# coding=utf-8
"""
Complexity model for vLLM inference.

Complexity is a decoder-only transformer with INL (Inertial Navigation Layer) dynamics
for numerical stability and smooth token generation.

Key innovations:
- INL Dynamics: PID-like control with velocity tracking (alpha, beta, gate, mu)
- Token-Routed MLP: Deterministic expert routing (token_id % num_experts)
- Mu-Guided Attention: Top-down influence from previous layer's equilibrium

Architecture:
    For each layer:
    1. Mu-Guided Attention (perception)
    2. INL Dynamics (control/stabilization)
    3. Token-Routed MLP (transformation)

Paper: [TODO: Add paper link]
GitHub: https://github.com/Complexity-ML/complexity-inference
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors


# =============================================================================
# Mu Dynamics Functions
# =============================================================================

def soft_clamp(x: torch.Tensor, min_val: float = -10.0, max_val: float = 10.0) -> torch.Tensor:
    """
    Differentiable soft clamping using tanh.
    Avoids hard discontinuities in gradients.
    """
    center = (max_val + min_val) / 2
    scale = (max_val - min_val) / 2
    return center + scale * torch.tanh((x - center) / scale)


def mu_clamp(
    x: torch.Tensor,
    mu: torch.Tensor,
    max_deviation: float = 5.0,
) -> torch.Tensor:
    """
    Soft clamp deviation from equilibrium point mu.

    Args:
        x: Input tensor
        mu: Equilibrium point (same shape as x or broadcastable)
        max_deviation: Maximum allowed deviation from mu

    Returns:
        Clamped tensor where |x - mu| <= max_deviation (soft)
    """
    deviation = x - mu
    clamped_deviation = soft_clamp(deviation, -max_deviation, max_deviation)
    return mu + clamped_deviation


# =============================================================================
# INL Dynamics
# =============================================================================

class INLDynamics(nn.Module):
    """
    INL (Inertial Navigation Layer) Dynamics for numerical stability.

    Implements PID-like control with velocity tracking:
        error = h - mu(h)                   # deviation from equilibrium
        v_next = alpha * v - beta * error   # velocity update
        h_next = h + dt * gate * v_next     # position update

    Parameters:
        - alpha: inertia/momentum (learnable, per-dim)
        - beta: correction strength (learnable, per-dim)
        - gate: amplitude control (learnable, per-dim)
        - mu: equilibrium point (learnable + contextual projection)
        - dt: integration timestep (fixed)
    """

    def __init__(
        self,
        hidden_size: int,
        controller_hidden: int = 64,
        dt: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        # Learnable equilibrium (base + contextual)
        self.mu = nn.Parameter(torch.zeros(hidden_size))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

        # Controller MLP: [h, v] -> alpha, beta, gate
        self.controller_in = nn.Linear(hidden_size * 2, controller_hidden)
        self.controller_out = nn.Linear(controller_hidden, hidden_size * 3)

        # Initialize for stable dynamics
        with torch.no_grad():
            bias = self.controller_out.bias
            bias[:hidden_size].fill_(2.2)           # alpha ~ 0.9
            bias[hidden_size:hidden_size*2].fill_(-2.2)  # beta ~ 0.1
            bias[hidden_size*2:].fill_(0.0)         # gate ~ 0.5
            self.controller_out.weight.normal_(0, 0.01)

    def forward(
        self,
        h: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply dynamics update.

        Args:
            h: Hidden states [num_tokens, hidden_size]
            v: Velocity states [num_tokens, hidden_size]

        Returns:
            h_next, v_next, mu_contextual
        """
        if v is None:
            v = torch.zeros_like(h)

        # Controller
        hv = torch.cat([h, v], dim=-1)
        ctrl = F.silu(self.controller_in(hv))
        ctrl_out = self.controller_out(ctrl)

        alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.hidden_size, dim=-1)
        alpha = torch.sigmoid(alpha_raw)
        beta = torch.clamp(F.softplus(beta_raw), max=2.0)
        gate = torch.sigmoid(gate_raw)

        # Contextual mu
        mu_contextual = self.mu + self.mu_proj(h)

        # Dynamics
        error = h - mu_contextual
        v_next = alpha * v - beta * error
        v_next = torch.clamp(v_next, min=-10.0, max=10.0)
        h_next = h + self.dt * gate * v_next

        return h_next, v_next, mu_contextual


class ComplexityMLP(nn.Module):
    """Standard MLP with SiLU activation (fallback when not using Token-Routed)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class TokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP - Deterministic expert routing with optional mu-guidance.

    Routes tokens to experts based on: expert_id = token_id % num_experts
    With mu-guidance: expert_id can be influenced by mu through mu_router.

    No router network needed (deterministic), no load balancing loss.

    Weight format (matches complexity-inference exactly):
    - intermediate_size is TOTAL, divided by num_experts for per-expert size
    - Uses fused gate_up_proj: [num_experts, hidden_size, 2 * expert_intermediate]
    - down_proj: [num_experts, expert_intermediate, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        vocab_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        # Per-expert intermediate size (total intermediate / num_experts)
        self.expert_intermediate_size = intermediate_size // num_experts

        # Fused gate+up projection: [num_experts, hidden_size, 2 * expert_intermediate]
        # First half is gate, second half is up (matches complexity-inference)
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * self.expert_intermediate_size)
        )
        # Down projection: [num_experts, expert_intermediate, hidden_size]
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, self.expert_intermediate_size, hidden_size)
        )

        # Mu-guided routing (optional override of token-based routing)
        self.mu_router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.mu_router.weight)  # Start neutral

        # Token to expert mapping buffer
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )

        # Initialize expert weights
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, hidden_size]
            token_ids: [num_tokens] - used for deterministic routing
            mu: [num_tokens, hidden_size] - optional mu for guided routing

        Routing logic (matches complexity-inference exactly):
            1. Base routing: expert_id = token_id % num_experts
            2. If mu provided: mu_logits bias the selection (hard argmax)
        """
        num_tokens = x.shape[0]

        # Determine expert assignment
        if token_ids is None:
            expert_ids = torch.zeros(num_tokens, dtype=torch.long, device=x.device)
        else:
            token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
            base_expert_ids = self.token_to_expert.to(x.device)[token_ids_clamped]

            # Mu-guided routing (matches complexity-inference)
            if mu is not None:
                mu_logits = self.mu_router(mu)  # [num_tokens, num_experts]
                # Combine: strong base routing (10.0) + mu influence
                base_one_hot = F.one_hot(base_expert_ids, self.num_experts).float()
                combined_logits = base_one_hot * 10.0 + mu_logits
                expert_ids = combined_logits.argmax(dim=-1)
            else:
                expert_ids = base_expert_ids

        # Gather weights for each token's expert
        gate_up_weights = self.gate_up_proj[expert_ids]  # [num_tokens, H, 2*I]
        down_weights = self.down_proj[expert_ids]  # [num_tokens, I, H]

        # Fused gate+up matmul
        gate_up_out = torch.bmm(x.unsqueeze(1), gate_up_weights).squeeze(1)  # [num_tokens, 2*I]

        # Split and apply SwiGLU
        gate_out = gate_up_out[..., :self.expert_intermediate_size]
        up_out = gate_up_out[..., self.expert_intermediate_size:]
        intermediate = F.silu(gate_out) * up_out  # [num_tokens, I]

        # Down projection
        output = torch.bmm(intermediate.unsqueeze(1), down_weights).squeeze(1)  # [num_tokens, H]

        return output


class ComplexityAttention(nn.Module):
    """
    Complexity attention with mu-guidance.

    Supports:
    - Grouped Query Attention (GQA)
    - RoPE positional embeddings
    - QK normalization
    - Mu-guided attention via learned projections (mu_to_q, mu_to_k, mu_to_v)

    Mu-guidance (INL 2025):
        Q = W_q @ x + mu_to_q @ mu_prev
        K = W_k @ x + mu_to_k @ mu_prev
        V = W_v @ x + mu_to_v @ mu_prev

    This creates top-down information flow where mu from layer N
    influences attention in layer N+1.
    """

    def __init__(
        self,
        config: ComplexityConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Mu-guided projections (INL 2025)
        # These project mu from previous layer to bias Q, K, V
        self.mu_to_q = nn.Linear(hidden_size, self.q_size, bias=False)
        self.mu_to_k = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.mu_to_v = nn.Linear(hidden_size, self.kv_size, bias=False)

        # Initialize mu projections to small values (start with minimal influence)
        for proj in [self.mu_to_q, self.mu_to_k, self.mu_to_v]:
            nn.init.normal_(proj.weight, std=0.01)

        # Output projection
        self.o_proj = RowParallelLinear(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
        )

        # QK Norm (optional)
        self.use_qk_norm = getattr(config, "use_qk_norm", True)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Attention backend
        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            positions: [num_tokens]
            hidden_states: [num_tokens, hidden_size]
            attn_metadata: vLLM attention metadata
            mu_prev: [num_tokens, hidden_size] - mu from previous layer (optional)
        """
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply mu-guidance via learned projections (INL 2025)
        if mu_prev is not None:
            q = q + self.mu_to_q(mu_prev)
            k = k + self.mu_to_k(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        # QK Norm
        if self.use_qk_norm:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(-1, self.q_size)
            k = k.view(-1, self.kv_size)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v, attn_metadata)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class ComplexityDecoderLayer(nn.Module):
    """
    Complexity decoder layer with INL dynamics.

    Architecture:
        1. Mu-Guided Attention (perception)
        2. INL Dynamics (control/stabilization)
        3. Token-Routed MLP (transformation)
    """

    def __init__(
        self,
        config: ComplexityConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_token_routed_mlp = getattr(config, "use_token_routed_mlp", True)

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention
        self.self_attn = ComplexityAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )

        # INL Dynamics
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            controller_hidden=getattr(config, "dynamics_controller_hidden", 64),
            dt=getattr(config, "dynamics_dt", 0.1),
        )

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP
        if self.use_token_routed_mlp:
            self.mlp = TokenRoutedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=getattr(config, "num_experts", 4),
                vocab_size=config.vocab_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = ComplexityMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        velocity_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        token_ids: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden_states, velocity_states, mu_current
        """
        # 1. Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            mu_prev=mu_prev,
        )

        # 2. INL Dynamics
        hidden_states, velocity_states, mu_current = self.dynamics(
            hidden_states, velocity_states
        )
        hidden_states = residual + hidden_states

        # 3. MU-GUIDED MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_token_routed_mlp:
            # Pass mu_current to enable mu-guided expert routing
            hidden_states = self.mlp(hidden_states, token_ids=token_ids, mu=mu_current)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, velocity_states, mu_current


class ComplexityModel(nn.Module):
    """Complexity transformer model (decoder-only)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            ComplexityDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [num_tokens]
            positions: [num_tokens]
            attn_metadata: vLLM attention metadata
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Initialize velocity
        velocity_states = torch.zeros_like(hidden_states)

        # Process layers with mu propagation
        mu_prev = None
        mu_residual = None

        for layer in self.layers:
            hidden_states, velocity_states, mu_current = layer(
                positions=positions,
                hidden_states=hidden_states,
                velocity_states=velocity_states,
                attn_metadata=attn_metadata,
                token_ids=input_ids,
                mu_prev=mu_prev,
            )

            # Mu residual highway
            if mu_residual is None:
                mu_residual = mu_current.clone()
            else:
                mu_residual = mu_residual + mu_current
            mu_prev = mu_current + 0.1 * mu_residual

        # Final norm
        hidden_states = self.norm(hidden_states)
        return hidden_states


class ComplexityForCausalLM(nn.Module):
    """
    Complexity model for causal language modeling.

    vLLM-compatible implementation with:
    - INL Dynamics for numerical stability
    - Token-Routed MLP for efficient expert routing
    - Mu-guided attention for top-down information flow
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
    ]

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        # Model
        self.model = ComplexityModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model",
        )

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.lm_head",
            )

        # Logits processor
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Sampler
        self.sampler = get_sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load weights from checkpoint."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            # Handle Token-Routed MLP weights
            if "mlp.gate_proj" in name or "mlp.up_proj" in name or "mlp.down_proj" in name:
                # These are expert weights, load directly
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue

            # Handle stacked params (QKV, gate_up)
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Regular params
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
