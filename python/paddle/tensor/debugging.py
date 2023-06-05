#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define functions to debug a tensor

from paddle import _C_ops
from paddle.amp.debugging import DebugMode

from ..framework import LayerHelper, in_dynamic_mode

__all__ = ["check_numerics"]


def check_numerics(
    tensor, op_type, var_name, debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT
):
    """
    This function is used to debugging a tensor, finding the number of NaNs, Infs and zeros in the tensor.

    Args:
        tensor(Tensor): The target tensor to check.
        op_type(str): The OP or API name which produce the target tensor.
        var_name(str): The name of target tensor.
        debug_mode(paddle.amp.debugging.DebugMode, optional): The mode of debugging to be used. Default is DebugMode.CHECK_NAN_INF_AND_ABORT.

    Returns:
        stats(Tensor): The output stats tensor stores the number of NaNs, Infs and zeros of input tensor. The shape is [3] and dtype is int64.
        values(Tensor): The output values tensor stores the maximum, minimum and mean value of input tensor. The shape is [3] and dtype is float.

    Examples:

        ..  code-block:: python

            import paddle

            checker_config = paddle.amp.debugging.TensorCheckerConfig(
                enable=True, debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF)

            x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32')
            y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
            res = paddle.pow(x, y)
            paddle.tensor.debugging.check_numerics(res, "pow", "res")

    """
    stack_height_limit = -1
    output_dir = ""

    if in_dynamic_mode():
        return _C_ops.check_numerics(
            tensor,
            op_type,
            var_name,
            debug_mode.value,
            stack_height_limit,
            output_dir,
        )

    helper = LayerHelper("check_numerics", **locals())

    stats = helper.create_variable_for_type_inference(dtype="int64")
    values = helper.create_variable_for_type_inference(dtype="float")

    helper.append_op(
        type='check_numerics',
        inputs={
            'Tensor': tensor,
        },
        attrs={
            'op_type': op_type,
            'var_name': var_name,
            'check_nan_inf_level': debug_mode.value,
            'stack_height_limit': stack_height_limit,
            'output_dir': output_dir,
        },
        outputs={'Stats': [stats], 'Values': [values]},
    )
    return stats, values
