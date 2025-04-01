# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch


@torch.compile
def normalize_output(q, k, o, cum_k=None):
    k = k.cumsum(-2)
    if cum_k is not None:
        k = k + cum_k
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-10)
