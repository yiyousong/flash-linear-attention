# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


@torch.jit.script
def normalize_output(
    q: torch.Tensor,
    k: torch.Tensor,
    o: torch.Tensor,
    z_state: torch.Tensor | None = None,
):
    # q, k, o: [B, T, H, *]; optional z_state: [B, 1, H, K] (running cumulative K).
    k_cum = k.cumsum(1)
    if z_state is not None:
        k_cum = k_cum + z_state
    z = (q * k_cum).sum(-1, keepdim=True)
    return o / (z + 1e-10), k_cum[:, -1:]
