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
from paddle import _C_ops, _legacy_C_ops

__all__ = ['custom_fused_dense', 'CustomFusedDense']



    # const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    # const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    # const phi::DenseTensor* bias = ctx.Input<phi::DenseTensor>("Bias");

    # phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    # phi::DenseTensor* reserve_space =
    #     ctx.Output<phi::DenseTensor>("ReserveSpace");

    # bool trans_x = ctx.Attr<bool>("trans_x");
    # bool trans_y = ctx.Attr<bool>("trans_y");

    # std::string activation = ctx.Attr<std::string>("activation");


def custom_fused_dense(x,
                      y,
                      bias,
                      transx,
                      transy,
                      use_addto):
    if _non_static_mode():
        return _legacy_C_ops.custom_fused_dense(x, y, bias, out, nullptr, 
                                            transx, transy, "linear")

    helper = LayerHelper('custom_fused_dense', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='custom_fused_dense',
                     inputs={
                         'X': x,
                         'Y': y,
                         'Bias': bias
                     },
                     outputs={
                         'Out': out,
                         'reserve_space': 'nullptr'
                    },
                     attrs={
                         'transx': transx,
                         'transy': transy,
                         'use_addto': use_addto
                     })
    return out

