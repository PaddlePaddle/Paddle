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

from paddle import _C_ops
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper


def int_bincount(x, low, high, dtype=None, name=None):
    if in_dynamic_or_pir_mode():
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
