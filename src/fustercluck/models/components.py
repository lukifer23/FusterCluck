"""Core transformer components for FusterCluck-450."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RotaryConfig:
    dim: int
    base: int = 10000
    scale_base: Optional[float] = None


class RMSNorm(nn.Module):
    """Root mean square layer normalization with optional epsilon scaling."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.size(-1)))
        x_norm = x / torch.maximum(rms, torch.tensor(self.eps, device=x.device, dtype=x.dtype))
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    """Implements rotary position embeddings (RoPE) for even head dimensions."""

    def __init__(self, cfg: RotaryConfig) -> None:
        super().__init__()
        if cfg.dim % 2 != 0:
            raise ValueError("Rotary dimension must be even.")
        inv_freq = 1.0 / (cfg.base ** (torch.arange(0, cfg.dim, 2, dtype=torch.float32) / cfg.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scale_base = cfg.scale_base

    def get_embed(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.scale_base is not None:
            scale = (torch.arange(seq_len, device=device, dtype=freqs.dtype) + 1.0)[..., None]
            freqs = freqs / (scale ** self.scale_base)
        freqs = freqs.to(dtype)
        return freqs.sin(), freqs.cos()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        sin, cos = self.get_embed(seq_len, q.device, q.dtype)
        q_rot, q_pass = torch.chunk(q, 2, dim=-1)
        k_rot, k_pass = torch.chunk(k, 2, dim=-1)
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]
        q_rotated = q_rot * cos + rotate_half(q_rot) * sin
        k_rotated = k_rot * cos + rotate_half(k_rot) * sin
        q = torch.cat((q_rotated, q_pass), dim=-1)
        k = torch.cat((k_rotated, k_pass), dim=-1)
        return q, k


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class GQAttention(nn.Module):
    """Grouped-query attention using PyTorch SDPA with RoPE support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope: Optional[RotaryEmbedding] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Embedding dim must be divisible by number of heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for rotary embeddings")
        self.rope = rope
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * num_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Expand KV heads to match number of query heads
        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.o_proj(attn_output)


class SwiGLUFeedForward(nn.Module):
    """Feed-forward network using SwiGLU gating."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        return self.down_proj(self.up_proj(x) * gate)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float,
        rope: Optional[RotaryEmbedding],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = GQAttention(dim, num_heads, num_kv_heads, rope, dropout)
        self.ffn = SwiGLUFeedForward(dim, hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attn_mask, is_causal=is_causal)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FusterCluckDecoder(nn.Module):
    """Decoder-only transformer with grouped-query attention and RoPE."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        mlp_ratio: float = 4.0,
        rope_theta: int = 10000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        rope_cfg = RotaryConfig(dim=dim // num_heads, base=rope_theta)
        rope = RotaryEmbedding(rope_cfg)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, num_kv_heads, mlp_ratio, rope, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeddings: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        if vision_embeddings is not None:
            if vision_mask is None:
                raise ValueError("vision_mask is required when vision_embeddings are provided")
            bsz = x.size(0)
            num_queries = vision_embeddings.size(1)
            for batch_idx in range(bsz):
                positions = vision_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    continue
                count = min(positions.numel(), num_queries)
                x[batch_idx, positions[:count]] = vision_embeddings[batch_idx, :count]
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Simple greedy sampling loop for quick smoke tests."""

        generated = [input_ids]
        cur_ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(cur_ids)[:, -1, :] / max(temperature, 1e-5)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            generated.append(next_token)
        return torch.cat(generated, dim=1)
