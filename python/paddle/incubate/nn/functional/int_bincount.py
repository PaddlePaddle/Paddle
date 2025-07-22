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

import paddle
from paddle import _C_ops
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper


def math_int_bincount(x, low, high, dtype):
    """
    A mathematically equivalent implementation of int_bincount using scatter and sum

    Args:
        x (Tensor): A 1D or 2D int64 tensor containing category indices.
        low (int): The minimum possible category index (usually 0).
        high (int): One past the maximum category index (i.e., number of categories).
        dtype (paddle.dtype): Data type of the output tensor (e.g., paddle.int64).

    Returns:
        Tensor: A 1D tensor of shape [high - low], where each element is
                the count of occurrences of that category in `x`.
    """
    if x.ndim not in [0, 1, 2]:
        raise ValueError(
            f"x must be a 0D, 1D or 2D tensor, but got ndim={x.ndim}"
        )

    if x.dtype not in [paddle.int32, paddle.int64]:
        raise ValueError(f"x.dtype must be int32 or int64, but got {x.dtype}")

    if dtype not in ['int32', 'int64', paddle.int32, paddle.int64]:
        raise ValueError(f"dtype must be 'int32' or 'int64', but got '{dtype}'")

    if high < low:
        raise ValueError(
            f"'high' ({high}) must be greater than or equal to 'low' ({low})"
        )

    if x.numel().item() == 0:
        return paddle.zeros([high - low], dtype=dtype)

    if x.ndim == 0:
        x = x.reshape([-1]).unsqueeze(0)  # Shape: [1, N]
    elif x.ndim == 1:
        x = x.unsqueeze(0)  # Shape: [1, N]

    x_min = x.min().item()
    x_max = x.max().item()

    if x_min < 0:
        raise ValueError(
            f"Elements of x must be non-negative, but got min={x_min}"
        )

    max_val = max(x_max + 1, high)
    mask = paddle.zeros([x.shape[0], max_val], dtype=x.dtype)
    mask = mask.put_along_axis(
        x, paddle.to_tensor(1.0, dtype=x.dtype), axis=1, reduce='add'
    )

    count = paddle.sum(mask, axis=0).cast(dtype)
    return count[low:high]


def int_bincount(x, low, high, dtype=None, name=None):
    if in_dynamic_or_pir_mode():
        if paddle.is_compiled_with_xpu():
            return math_int_bincount(x, low, high, dtype)
        else:
            return _C_ops.int_bincount(x, low, high, dtype)

    helper = LayerHelper("int_bincount", **locals())
    out_dtype = dtype if dtype is not None else x.dtype
    y = helper.create_variable_for_type_inference(dtype=out_dtype)
    dtype_attr = convert_dtype(out_dtype)

    helper.append_op(
        type="int_bincount",
        inputs={"x": x},
        outputs={"y": y},
        attrs={
            "low": low,
            "high": high,
            "dtype": dtype_attr,
        },
    )
    return y
