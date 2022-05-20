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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode
from paddle.tensor.linalg import matmul
from paddle import _C_ops


def fused_matmul_bias(x,
                      y,
                      bias=None,
                      transpose_x=False,
                      transpose_y=False,
                      name=None):
    if bias is None:
        return matmul(x, y, transpose_x, transpose_y, name)
    if _non_static_mode():
        return _C_ops.fused_gemm_epilogue(x, y, bias, 'trans_x', transpose_x,
                                          'trans_y', transpose_y)

    helper = LayerHelper('fused_matmul_bias', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fused_gemm_epilogue',
        inputs={'X': x,
                'Y': y,
                'Bias': bias},
        outputs={'Out': out},
        attrs={'trans_x': transpose_x,
               'trans_y': transpose_y})
    return out


def fused_linear(x, weight, bias=None, transpose_weight=False, name=None):
    return fused_matmul_bias(x, weight, bias, False, transpose_weight, name)
