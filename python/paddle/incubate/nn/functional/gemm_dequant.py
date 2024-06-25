# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode


def gemm_dequant(x, y, dequant_out_scales, bfloat16_out=True):
    if in_dynamic_or_pir_mode():
        return _C_ops.gemm_dequant(x, y, dequant_out_scales, bfloat16_out)

    helper = LayerHelper('gemm_dequant', **locals())
    if bfloat16_out:
        out = helper.create_variable_for_type_inference(dtype="bfloat16")
    else:
        out = helper.create_variable_for_type_inference(dtype="float16")

    inputs = {}
    inputs['x'] = x
    inputs['y'] = y
    inputs['dequant_out_scales'] = dequant_out_scales

    outputs = {
        'out': out,
    }
    helper.append_op(
        type='gemm_dequant',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'bfloat16_out': bfloat16_out,
        },
    )
    return out
