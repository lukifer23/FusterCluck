"""Vision encoder and fusion modules."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
except ImportError as exc:  # pragma: no cover - handled in docs
    raise RuntimeError("torchvision>=0.18.0 is required for vision modules") from exc


class VisionBackbone(nn.Module):
    """Wraps a ViT-B/16 backbone with optional projection."""

    def __init__(self, image_size: int = 224, trainable: bool = False) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
        if image_size != 224:
            model.patch_size = (16, 16)
            model.image_size = (image_size, image_size)
        for p in model.parameters():
            p.requires_grad = trainable
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns encoded tokens (CLS + patches) from the ViT backbone."""

        if images.dtype != torch.float32:
            images = images.float()
        x = self.model._process_input(images)
        n = x.shape[0]
        x = x.reshape(n, self.model.num_patches, -1)
        cls_token = self.model.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.encoder(x)
        return x

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Returns pooled CLS representations."""

        tokens = self.forward(images)
        return tokens[:, 0]


class QFormerMini(nn.Module):
    """Lightweight Q-Former inspired module for cross-attention with learnable queries."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_layers: int,
        num_queries: int = 64,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        bsz = vision_tokens.size(0)
        queries = self.query_tokens.expand(bsz, -1, -1)
        attn_output, _ = self.cross_attn(queries, vision_tokens, vision_tokens, need_weights=False)
        fused = self.layers(attn_output)
        return self.norm(fused)


class PerceiverResampler(nn.Module):
    """Downsamples vision tokens to a fixed number of latents via cross attention."""

    def __init__(
        self,
        dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(dim, num_heads, batch_first=True),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "norm": nn.LayerNorm(dim),
                    }
                )
            )

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        bsz = vision_tokens.size(0)
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](latents, vision_tokens, vision_tokens, need_weights=False)
            latents = latents + attn_out
            latents = latents + layer["ffn"](latents)
            latents = layer["norm"](latents)
        return latents


class VisionProjector(nn.Module):
    """Projects fused vision tokens into the text model hidden space."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(fused_tokens)


class MultimodalAdapter(nn.Module):
    """Combines vision backbone, fusion, and projection layers."""

    def __init__(
        self,
        hidden_dim: int,
        fusion: str = "qformer",
        num_queries: int = 64,
        trainable_backbone: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.backbone = VisionBackbone(image_size=image_size, trainable=trainable_backbone)
        vision_dim = self.backbone.model.hidden_dim
        if fusion == "qformer":
            self.fusion = QFormerMini(
                hidden_size=vision_dim,
                num_attention_heads=8,
                intermediate_size=vision_dim * 4,
                num_layers=3,
                num_queries=num_queries,
            )
        elif fusion == "perceiver":
            self.fusion = PerceiverResampler(dim=vision_dim, num_latents=num_queries, num_heads=8, num_layers=3)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")
        self.projector = VisionProjector(vision_dim, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward(images)
        fused = self.fusion(tokens)
        return self.projector(fused)
