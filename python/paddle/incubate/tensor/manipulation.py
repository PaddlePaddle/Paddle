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
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode

__all__ = []


# TODO(qili93): remove this op after custom op and custom device
# integrated and then move this op along with its code to plugin.
def _npu_identity(x, format=-1):
    """

    This OP takes in the Tensor :attr:`x` and change it to output with
    aclFormat with int value. This API is only used for Ascend NPU.

    Args:
        x(Tensor): An input N-D Tensor with data type bool, float16,
                   float32, float64, int32, int64, int16, int8, uint8.
        format(int): Storage data format of the output in aclFormat,
                     default value is -1.

    Returns:
        Tensor: A Tensor with acl storage format on Ascend NPU.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:NPU)
            >>> import paddle
            >>> paddle.device.set_device('npu')

            >>> x = paddle.ones(shape=[6])
            >>> y = paddle.incubate._npu_identity(x, 3) # ACL_FORMAT_NC1HWC0 = 3
            >>> print(y.shape)
            [1, 1, 1, 1, 16]
    """
    if in_dynamic_mode():
        return _C_ops.npu_identity(x, format)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'int8',
                'uint8',
                'int16',
                'int32',
                'int64',
                'float16',
                'float32',
                'float64',
            ],
            'npu_identity',
        )

        helper = LayerHelper('npu_identity', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=x.stop_gradient
        )
        helper.append_op(
            type='npu_identity',
            inputs={'x': [x]},
            outputs={'out': [out]},
            attrs={'format': format},
        )
        return out
