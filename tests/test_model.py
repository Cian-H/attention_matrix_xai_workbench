# tests/test_model.py
"""Tests for the MultiheadPooledAttention module.

The test checks that the module can be instantiated with the default
parameters and that a forward pass produces an output of the expected shape.
"""

from __future__ import annotations

import torch
import pytest

# Import the public API re‑exported in the package’s __init__.
from attention_matrix_xai_workbench import MultiheadPooledAttention


@pytest.mark.parametrize(
    "batch, seq_len, embed_dim, num_queries",
    [
        (2, 16, 8, 4),   # basic case
        (1, 32, 16, 6),  # larger embedding
    ],
)
def test_multihead_pooled_attention_forward(
    batch: int, seq_len: int, embed_dim: int, num_queries: int
) -> None:
    """Run a forward pass and verify output shape."""
    torch.manual_seed(0)

    # Random input tensor: [batch, seq_len, embed_dim]
    x = torch.randn(batch, seq_len, embed_dim)

    # Instantiate the module (using the default “softmax” activation)
    attn = MultiheadPooledAttention(
        embed_dim=embed_dim,
        num_queries=num_queries,
        attn_activation="softmax",
    )

    # Forward pass – should not raise any exception
    out = attn(x)

    # Expected shape: [batch, embed_dim] (collapsed across queries)
    assert out.shape == (batch, embed_dim), (
        f"Expected output shape {(batch, embed_dim)}, got {out.shape}"
    )
    # Output should be a floating‑point tensor
    assert out.dtype.is_floating_point, "Output tensor must be floating point"
