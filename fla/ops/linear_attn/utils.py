# -*- coding: utf-8 -*-

import torch


@torch.jit.script
def normalize_output(q, k, o, cum_k=None):
    k = k.cumsum(-2)
    if cum_k is not None:
        k=k+cum_K
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-10)
