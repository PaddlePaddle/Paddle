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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode
from paddle.fluid import core
from paddle import _C_ops, _legacy_C_ops


def softmax_mask_fuse(x, mask, name=None):
    """
    Do a masked softmax on x.

    This is designed for speeding up Transformer structure.
    Used for reducing operation such as: tmp = x + mask, out = softmax(tmp).
    The equation is:

    .. math::
        out = softmax(x + mask)

    **Note**:
        This API only supports GPU.

    Args:
        x (4-D Tensor): The input tensor, should be in 4D shape, it's data type should be float16, float32.
                        The fourth dimension of x must be larger or equal to 32 and less then 8192.
        mask (4-D Tensor): The input tensor, should be in 4D shape, it's data type should be float16, float32.
                           The second dimension of mask must be 1, and other dimensions must be same with x.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        4-D Tensor. A location into which the result is stored. It’s dimension is 4D. Has same shape with x.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            import paddle.incubate as incubate

            x = paddle.rand([2, 8, 8, 32])
            mask = paddle.rand([2, 1, 8, 32])

            rst = incubate.softmax_mask_fuse(x, mask)
            # [[[[0.02404429, 0.04658398, 0.02746007, ..., 0.01489375, 0.02397441, 0.02851614] ... ]]]
    """
    if _non_static_mode():
        out = _legacy_C_ops.fused_softmax_mask(x, mask)
        return out
    helper = LayerHelper('fused_softmax_mask', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='fused_softmax_mask',
                     inputs={
                         'X': [x],
                         'Mask': [mask]
                     },
                     outputs={'Out': [out]})
    return out
