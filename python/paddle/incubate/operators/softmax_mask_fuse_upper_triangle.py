# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from paddle import _legacy_C_ops
from paddle.fluid.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode


def softmax_mask_fuse_upper_triangle(x):
    """
    Do a masked softmax on x, which will always mask upper triangle part of x.

    This is designed for speeding up GPT kind Transformer structure.
    Used for reducing operation such as: tmp = x + mask, out = softmax(tmp), where the mask is
    always be an upper triangle matrix.
    The equation is:

    .. math::
        out = softmax(LowerTriangular(x))

    Note:
        This API only supports GPU.

    Args:
        x (4-D Tensor): The input tensor, should be in 4D shape, it's data type should be float16, float32
                        The fourth dimension of x must be larger or equal to 32 and less then 8192.
                        The third dimension of x must be same with the fourth dimension of x.

    Returns:
        4-D Tensor. A location into which the result is stored. It’s dimension is 4D. Has same dimension with x.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            import paddle.incubate as incubate

            x = paddle.rand((1, 1, 32, 32))

            rst = incubate.softmax_mask_fuse_upper_triangle(x)
            # [[[[1.        , 0.        , 0.        , ..., 0., 0., 0.],
            #    [0.45324376, 0.54675621, 0.        , ..., 0., 0., 0.],
            #    [0.32674268, 0.28156221, 0.39169508, ..., 0., 0., 0.]
            #     ... ]]]
    """
    if in_dynamic_mode():
        out = _legacy_C_ops.fused_softmax_mask_upper_triangle(x)
        return out

    helper = LayerHelper('fused_softmax_mask_upper_triangle', **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='fused_softmax_mask_upper_triangle',
        inputs={'X': [x]},
        outputs={'Out': [out]},
    )
    return out
