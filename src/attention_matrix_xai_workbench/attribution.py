from typing import TYPE_CHECKING, NamedTuple

from captum.attr import LayerIntegratedGradients

if TYPE_CHECKING:
    import torch


class AttentionContributions(NamedTuple):
    """Element-wise Attention × Gradient attributions for a single forward/backward pass.

    Both tensors are computed from the same ``backward()`` call, so no extra
    passes are needed to obtain both views.

    Attributes:
        attn_weights: Contribution of each post-softmax attention weight to the
            target output.  Shape ``[batch, num_heads, num_queries, seq_len]``.
            Captures *which input tokens* each query head was attending to and how
            much that attention contributed to the prediction.
        query_token: Contribution of each element of the learnable query parameter
            to the target output.  Shape ``[1, num_queries, embed_dim]``.
            Captures *what each latent query was looking for* — i.e. the direction
            in embedding space that drove the attention pattern.
    """

    attn_weights: "torch.Tensor"
    query_token: "torch.Tensor"


def layer_integrated_gradients(model, x, target_out_idx, attention_layer):
    """
    Calculates the contribution of the attention matrix to a specific output index
    using the Layer Integrated Gradients method.
    """
    model.eval()

    original_export_state = getattr(model, "export_attention", False)
    model.export_attention = True

    try:
        lig = LayerIntegratedGradients(model, attention_layer)
        attributions, delta = lig.attribute(
            inputs=x, target=target_out_idx, return_convergence_delta=True
        )
    finally:
        model.export_attention = original_export_state

    return attributions


def attention_x_gradient(
    model,
    x,
    target_out_idx: int,
    attention_layer_name: str = "attn_matrix_hookpoint",
) -> AttentionContributions:
    """Compute Attention × Gradient attributions for a specific output index.

    A forward hook captures the attention weight tensor as it passes through the
    named hookpoint module.  A tensor-level grad hook captures the corresponding
    gradient during the subsequent ``backward()`` call.  Because ``model.query_token``
    is a leaf ``nn.Parameter``, its ``.grad`` is populated by the same ``backward()``
    call at no extra cost.  The element-wise product of value × gradient gives a
    signed, per-element measure of contribution to the chosen output dimension.

    Args:
        model: The ``MultiheadPooledAttention`` model (or any model that exposes a
            sub-module whose name contains ``attention_layer_name`` and a
            ``query_token`` parameter).
        x: Input tensor of shape ``[batch, seq, d_model]``.
        target_out_idx: Index into the model's output vector to differentiate.
        attention_layer_name: Name (or substring) of the ``nn.Module`` to hook.
            Defaults to ``"attn_matrix_hookpoint"``.

    Returns:
        An :class:`AttentionContributions` named tuple with fields:

        * ``attn_weights`` — shape ``[batch, num_heads, num_queries, seq_len]``
        * ``query_token``  — shape ``[1, num_queries, embed_dim]``

    Raises:
        ValueError: If no module matching ``attention_layer_name`` is found, or if
            the model has no ``query_token`` parameter.
        RuntimeError: If no gradient flowed back through the hookpoint tensor.
    """
    model.eval()

    if not hasattr(model, "query_token"):
        raise ValueError(
            "model has no 'query_token' parameter.  "
            "Pass a MultiheadPooledAttention instance or set attention_layer_name appropriately."
        )

    _attn_live: list[torch.Tensor] = []   # live (grad-connected) tensor
    _attn_grad: list[torch.Tensor] = []   # gradient accumulated during backward

    def _forward_hook(module, inputs, outputs):
        outputs.retain_grad()
        _attn_live.append(outputs)
        outputs.register_hook(lambda grad: _attn_grad.append(grad))

    hook_handle = None
    for name, module in model.named_modules():
        if attention_layer_name in name:
            hook_handle = module.register_forward_hook(_forward_hook)
            break

    if hook_handle is None:
        raise ValueError(
            f"No module whose name contains '{attention_layer_name}' was found in the model."
        )

    model.zero_grad()
    output = model(x)

    # Differentiate the scalar output at target_out_idx w.r.t. all leaf tensors.
    # This populates:
    #   - _attn_grad  via the tensor-level hook on the hookpoint output
    #   - model.query_token.grad  automatically, because it is a leaf parameter
    target_val = output[0, target_out_idx]
    target_val.backward()

    hook_handle.remove()

    if not _attn_grad:
        raise RuntimeError(
            "No gradient was captured for the attention matrix.  Ensure that the "
            "tensor flowing through the hookpoint is connected to the computation "
            "graph (i.e. requires_grad=True somewhere upstream)."
        )

    if model.query_token.grad is None:
        raise RuntimeError(
            "model.query_token.grad is None after backward().  "
            "Ensure query_token requires_grad (it should by default as an nn.Parameter)."
        )

    attn_contrib = _attn_live[0].detach() * _attn_grad[0]
    qt_contrib = model.query_token.detach() * model.query_token.grad

    return AttentionContributions(attn_weights=attn_contrib, query_token=qt_contrib)
