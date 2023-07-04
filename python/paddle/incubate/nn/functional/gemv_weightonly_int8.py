# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper


def gemv_weightonly_int8(x, weight, bias, weight_scale, act_method):
    if in_dygraph_mode():
        return _C_ops.gemv_weightonly_int8(
            x, weight, bias, weight_scale, act_method
        )

    helper = LayerHelper('gemv_weightonly_int8', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='gemv_weightonly_int8',
        inputs={
            'x': x,
            'weight': weight,
            'bias': bias,
            "weight_scale": weight_scale,
        },
        attrs={"act_method": act_method},
        outputs={'out': out},
    )
    return out
