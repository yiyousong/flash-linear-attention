# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.simple_gla import chunk_simple_gla


@torch.compiler.disable
def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | tuple | None = None,
    output_final_state: bool = False,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        scale (Optional[float]):
            Scale factor for the linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor | tuple]):
            Initial recurrent state. When `normalize=False`, a tensor of shape `[B, H, K, V]`.
            When `normalize=True`, a tuple `(kv_state, z_state)` where `z_state` has shape
            `[B, 1, H, K]` (running cumulative key for the denominator). Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor | tuple | None):
            Mirrors the shape of `initial_state`: a tensor when `normalize=False`, a
            `(kv_state, z_state)` tuple when `normalize=True`, or `None` when
            `output_final_state=False`.
    """
    if q.shape[1] < q.shape[2]:
        raise DeprecationWarning(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...]. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
        )
    if normalize and isinstance(initial_state, tuple):
        kv_init, z_init = initial_state
    else:
        kv_init, z_init = initial_state, None
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=kv_init,
        output_final_state=output_final_state,
    )
    if normalize:
        k_cum = k.cumsum(1) if z_init is None else k.cumsum(1) + z_init
        o = o / ((q * scale * k_cum).sum(-1, keepdim=True) + 1e-10)
        return o, (final_state, k_cum[:, -1:])
    return o, final_state
