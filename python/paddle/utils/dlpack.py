# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import paddle

from ..base.core import LoDTensor
from ..base.data_feeder import check_type
from ..base.framework import in_dygraph_mode

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

    from paddle import Tensor

__all__ = [
    'to_dlpack',
    'from_dlpack',
]


class DLDeviceType(enum.IntEnum):
    # Enums as in DLPack specification (aten/src/ATen/dlpack.h)
    kDLCPU = (1,)
    kDLGPU = (2,)
    kDLCPUPinned = (3,)
    kDLOpenCL = (4,)
    kDLVulkan = (7,)
    kDLMetal = (8,)
    kDLVPI = (9,)
    kDLROCM = (10,)
    kDLExtDev = (12,)
    kDLOneAPI = (14,)


def to_dlpack(x: Tensor) -> CapsuleType:
    """
    Encodes a tensor to DLPack.

    Args:
        x (Tensor): The input tensor, and the data type can be `bool`, `float16`, `float32`,
                    `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`,
                    `complex128`.

    Returns:
        dltensor, and the data type is PyCapsule.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # x is a tensor with shape [2, 4]
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]])
            >>> dlpack = paddle.utils.dlpack.to_dlpack(x)
            >>> print(dlpack)
            >>> # doctest: +SKIP('the address will change in every run')
            <capsule object "dltensor" at 0x7f6103c681b0>
    """

    if in_dygraph_mode():
        if not isinstance(x, paddle.Tensor):
            raise TypeError(
                "The type of 'x' in to_dlpack must be paddle.Tensor,"
                f" but received {type(x)}."
            )

        return x.value().get_tensor()._to_dlpack()

    check_type(x, 'x', (LoDTensor), 'to_dlpack')
    return x._to_dlpack()


def from_dlpack(ext_tensor) -> Tensor:
    """
    Decodes a DLPack to a tensor.

    Args:
        dlpack (PyCapsule): a PyCapsule object with the dltensor.

    Returns:
        out (Tensor), a tensor decoded from DLPack. One thing to be noted, if we get
                      an input dltensor with data type as `bool`, we return the decoded
                      tensor as `uint8`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # x is a tensor with shape [2, 4]
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]])
            >>> dlpack = paddle.utils.dlpack.to_dlpack(x)
            >>> x = paddle.utils.dlpack.from_dlpack(dlpack)
            >>> print(x)
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[0.20000000, 0.30000001, 0.50000000, 0.89999998],
                    [0.10000000, 0.20000000, 0.60000002, 0.69999999]])
    """

    if hasattr(ext_tensor, '__dlpack__'):
        device = ext_tensor.__dlpack_device__()
        # device is either CUDA or ROCm, we need to pass the current
        # stream
        if device[0] in (DLDeviceType.kDLGPU, DLDeviceType.kDLROCM):
            stream = paddle.device.cuda.current_stream(device[1])
            # cuda_stream is the pointer to the stream and it is a public
            # attribute, but it is not documented
            # The array API specify that the default legacy stream must be passed
            # with a value of 1 for CUDA
            # https://data-apis.org/array-api/latest/API_specification/array_object.html?dlpack-self-stream-none#dlpack-self-stream-none
            is_gpu = device[0] == DLDeviceType.kDLGPU
            # Since pytorch is not using PTDS by default, lets directly pass
            # the legacy stream
            stream_ptr = (
                1 if is_gpu and stream.cuda_stream == 0 else stream.cuda_stream
            )
            dlpack = ext_tensor.__dlpack__(stream=stream_ptr)
        else:
            dlpack = ext_tensor.__dlpack__()
    else:
        # Old versions just call the converter
        dlpack = ext_tensor

    out: paddle.base.libpaddle.Tensor = paddle.base.core.from_dlpack(dlpack)

    if in_dygraph_mode():
        out: Tensor = paddle.to_tensor(out)

    return out
