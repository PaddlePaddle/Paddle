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


def softmax_mask_fuse(x, mask, name=None):
    """
    Do a masked softmax on x.

    This is designed for speeding up Transformer structure.
    Used for reducing operation such as: tmp = x + mask, out = softmax(tmp).
    The equation is:

    .. math::
        out = softmax(x + mask)

    Note:
        This API only supports GPU.

    Args:
        x (4-D Tensor): The input tensor, should be in 4D shape, it's data type should be float16, float32.
                        The fourth dimension of x must be larger or equal to 32 and less then 8192.
        mask (4-D Tensor): The input tensor, should be in 4D shape, it's data type should be float16, float32.
                           The second dimension of mask must be 1, and other dimensions must be same with x.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        4-D Tensor. A location into which the result is stored. It's dimension is 4D. Has same shape with x.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> import paddle.incubate as incubate

            >>> x = paddle.rand([2, 8, 8, 32])
            >>> mask = paddle.rand([2, 1, 8, 32])

            >>> rst = incubate.softmax_mask_fuse(x, mask)
            >>> print(rst)
            Tensor(shape=[1, 1, 32, 32], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [[[[1.        , 0.        , 0.        , ..., 0.        ,
                0.        , 0.        ],
               [0.49575609, 0.50424391, 0.        , ..., 0.        ,
                0.        , 0.        ],
               [0.26035303, 0.25114325, 0.48850375, ..., 0.        ,
                0.        , 0.        ],
                ...,
               [0.04379999, 0.04194880, 0.05150032, ..., 0.02721255,
                0.        , 0.        ],
               [0.02348574, 0.01959674, 0.02609110, ..., 0.04046615,
                0.02248267, 0.        ],
               [0.02280738, 0.03144657, 0.02892209, ..., 0.03885521,
                0.03342311, 0.02842640]]]])
    """
    if in_dynamic_mode():
        out = _legacy_C_ops.fused_softmax_mask(x, mask)
        return out
    helper = LayerHelper('fused_softmax_mask', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fused_softmax_mask',
        inputs={'X': [x], 'Mask': [mask]},
        outputs={'Out': [out]},
    )
    return out
