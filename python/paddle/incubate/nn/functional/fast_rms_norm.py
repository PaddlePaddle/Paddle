# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor


def fast_rms_norm(
    x: Tensor,
    scale: Tensor,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    r"""
    Apply Fast LayerNorm kernel.

    Args:
        x (Tensor): the input Tensor..
        scale (Tensor): the weight Tensor to affine output.
        epsilon (float): a small float number to avoid divide 0.

    Returns:
        y: the Tensor after performing layernorm.
        invvar: the invert variance(scaling factor) of y
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fast_rms_norm(
            x,
            scale,
            epsilon,
        )
