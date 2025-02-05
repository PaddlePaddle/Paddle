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

import copy

import paddle
from paddle import _C_ops
from paddle.base import core
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.framework import EagerParamBase
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


def _load_reload_impl(src_tensor, func):
    if isinstance(src_tensor, EagerParamBase):
        state = copy.deepcopy(src_tensor.__dict__)
        new_param = EagerParamBase(src_tensor.shape, src_tensor.dtype, **state)
        task = func(new_param, src_tensor)
        return new_param, task
    elif isinstance(src_tensor, paddle.Tensor):
        new_varbase = core.eager.Tensor()
        task = func(new_varbase, src_tensor)
        return new_varbase, task


def create_async_load():
    """Constructs a new AsyncLoad object. It is used to load/reload data asynchronously."""
    return core.AsyncLoad()


def async_offload(src_tensor, async_load):
    """
    Loads the source tensor into the destination tensor asynchronously.

    Args:
        src_tensor (EagerParamBase|paddle.Tensor): The source tensor.
        async_load (core.AsyncLoad): The AsyncLoad object.

    Returns:
        tuple: A tuple containing two elements:
         - dest_tensor (EagerParamBase|paddle.Tensor): The destination tensor.
         - task (Task): The task that loads the source tensor into the destination tensor.
    """
    return _load_reload_impl(src_tensor, async_load.offload)


def async_reload(src_tensor, async_load):
    """
    Reloads the source tensor into the destination tensor asynchronously.

    Args:
        src_tensor (EagerParamBase|paddle.Tensor): The source tensor.
        async_load (core.AsyncLoad): The AsyncLoad object.

    Returns:
        tuple: A tuple containing two elements:
         - dest_tensor (EagerParamBase|paddle.Tensor): The destination tensor.
         - task (Task): The task that reloads the source tensor into the destination tensor.
    """
    return _load_reload_impl(src_tensor, async_load.reload)


def async_offload_with_offset(
    src_tensor, dst_tensor, src_offset, dst_offset, offload_size, async_loader
):
    """
    Offloading the source tensor into the destination tensor asynchronously with offset and size customized.

    Args:
        src_tensor (EagerParamBase|paddle.Tensor): The source tensor.
        dst_tensor (EagerParamBase|paddle.Tensor): The destination tensor.
        src_offset (int): The element offset of the source tensor.
        dst_offset (int): The element offset of the destination tensor.
        offload_size (int): The size of the data to be loaded.
        async_loader (core.AsyncLoad): The AsyncLoad object.

    Returns:
        task (Task): The task that operates partial offloading.
    """
    assert len(src_tensor.shape) <= 1, "Only support 1-D tensor"
    assert len(dst_tensor.shape) <= 1, "Only support 1-D tensor"
    assert src_tensor.dtype == dst_tensor.dtype, "Only support same dtype"
    return async_loader.offload_with_offset(
        dst_tensor, src_tensor, dst_offset, src_offset, offload_size
    )
