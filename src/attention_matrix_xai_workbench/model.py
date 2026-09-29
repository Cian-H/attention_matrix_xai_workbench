"Implementation of MPA, pulled from github.com/Cian-H/HyDRA"

from collections.abc import Callable
from typing import ClassVar

import torch
from torch import nn


class MultiheadPooledAttention(nn.Module):
    """Aggregates variable length inputs to fixed tokens via multihead cross-attention."""

    ACTIVATION_REGISTRY: ClassVar[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = {
        "softmax": nn.Softmax(dim=-1),
        "time-series": nn.Softmax(dim=-1),
        "linear": nn.Identity(),
        "relational": nn.Identity(),
    }

    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 4,
        attn_activation: str | Callable[[torch.Tensor], torch.Tensor] = "softmax",
        num_heads: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes MultiheadPooledAttention.

        Args:
            embed_dim: The number of features (or columns) in the input data.
            num_queries: The number of learnable latent vectors used to compress the sequence.
            attn_activation: The activation function applied to the attention weights.
            num_heads: The number of attention heads for the cross-attention mechanism.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        self.num_tokens = num_queries
        self.activation = attn_activation
        self.query_token = nn.Parameter(
            torch.randn(1, num_queries, embed_dim, device=device, dtype=dtype)
        )

        if num_heads is None:
            num_heads = 4 if embed_dim % 4 == 0 else (2 if embed_dim % 2 == 0 else 1)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, device=device, dtype=dtype
        )
        self.proj = nn.Linear(num_queries * embed_dim, embed_dim, device=device, dtype=dtype)

    def _get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Resolves string keys to activations, passing through callables directly."""
        if isinstance(self.activation, str):
            return self.ACTIVATION_REGISTRY.get(self.activation.lower(), nn.Identity())
        return self.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuses input sequence into a single context representation.

        Args:
            x: Input tensor of shape `[batch, seq, d_model]`.
        """
        batch_size = x.size(0)
        q = self.query_token.expand(batch_size, -1, -1).contiguous()
        x = x.contiguous()

        is_softmax = isinstance(self.activation, str) and self.activation.lower() in {
            "softmax",
            "time-series",
        }

        if is_softmax:
            backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]

            # Chunk along batch dimension to prevent exceeding CUDA grid block limit (65,535)
            max_chunk = min(4096, max(1, 65535 // self.attn.num_heads))
            if batch_size > max_chunk:
                out_chunks = []
                for i in range(0, batch_size, max_chunk):
                    q_c = q[i : i + max_chunk].contiguous()
                    x_c = x[i : i + max_chunk].contiguous()
                    try:
                        with torch.nn.attention.sdpa_kernel(backends):
                            out_c, _ = self.attn(q_c, x_c, x_c, need_weights=False)
                    except RuntimeError, torch.AcceleratorError:
                        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                            out_c, _ = self.attn(q_c, x_c, x_c, need_weights=False)
                    out_chunks.append(out_c)
                out = torch.cat(out_chunks, dim=0)
            else:
                try:
                    with torch.nn.attention.sdpa_kernel(backends):
                        out, _ = self.attn(q, x, x, need_weights=False)
                except RuntimeError, torch.AcceleratorError:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                        out, _ = self.attn(q, x, x, need_weights=False)
        else:
            # Manual multi-head attention to support custom activations
            B, T, D = q.shape
            _, S, _ = x.shape
            H = self.attn.num_heads
            D_h = D // H

            # Extract weights
            in_w = self.attn.in_proj_weight
            in_b = self.attn.in_proj_bias

            # Project Q, K, V
            q_proj = torch.nn.functional.linear(q, in_w[:D], in_b[:D] if in_b is not None else None)
            k_proj = torch.nn.functional.linear(
                x, in_w[D : 2 * D], in_b[D : 2 * D] if in_b is not None else None
            )
            v_proj = torch.nn.functional.linear(
                x, in_w[2 * D :], in_b[2 * D :] if in_b is not None else None
            )

            # Reshape for heads
            q_proj = q_proj.view(B, T, H, D_h).transpose(1, 2)  # [B, H, T, D_h]
            k_proj = k_proj.view(B, S, H, D_h).transpose(1, 2)  # [B, H, S, D_h]
            v_proj = v_proj.view(B, S, H, D_h).transpose(1, 2)  # [B, H, S, D_h]

            # Compute scaled dot-product attention logits
            attn_weights = torch.matmul(
                # [B, H, T, S]
                q_proj,
                k_proj.transpose(-2, -1),
            ) / (D_h**0.5)

            # Apply custom activation
            activation_fn = self._get_activation_fn()
            attn_weights = activation_fn(attn_weights)

            # Context fusion
            out = torch.matmul(attn_weights, v_proj)  # [B, H, T, D_h]
            out = out.transpose(1, 2).reshape(B, T, D)
            out = self.attn.out_proj(out)

        out = out.reshape(batch_size, self.num_tokens * x.size(-1))
        return self.proj(out)
