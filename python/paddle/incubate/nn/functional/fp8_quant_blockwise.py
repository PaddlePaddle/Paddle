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
from paddle.base.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor


def fp8_quant_blockwise(
    X: Tensor,
    epsilon: float = 0.0,
    input_transpose: bool = False,
    output_scale_transpose: bool = True,
    using_pow2_scale: bool = True,
    quant_method: str = "1x128",
    output_type: str = "e4m3",
    name: str | None = None,
):
    if quant_method == "1x128":
        using_1x128 = True
    elif quant_method == "128x128":
        using_1x128 = False
    else:
        raise ValueError("Unsupported quantization method")

    if output_type == "e4m3":
        using_e5m2 = False
    else:
        raise ValueError("Unsupported output type")

    if in_dynamic_or_pir_mode():
        X_fp8, scale, X_fp8_t, scale_t = (
            _C_ops.fp9_quant_blockwise(
                X,
                using_1x128,
                input_transpose,
                output_scale_transpose,
                using_e5m2,
                using_pow2_scale,
            ),
            None,
        )
        # Aligned with kitchen's logic
        if not input_transpose:
            return X_fp8, scale
        else:
            return X_fp8, scale, X_fp8_t, scale_t
