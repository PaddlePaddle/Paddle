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
import warnings
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

import paddle

from ..base.core import DenseTensor
from ..base.data_feeder import check_type
from ..base.framework import in_dygraph_mode

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

    from paddle import Tensor


__all__ = [
    'to_dlpack',
    'from_dlpack',
]

_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportDLPack(Protocol[_T_contra]):
    """
    ref:
        https://github.com/numpy/numpy/blob/7e6e48ca7aacae9994d18a3dadbabd2b91c32151/numpy/__init__.pyi#L3068-L3077
        https://github.com/numpy/numpy/blob/7e6e48ca7aacae9994d18a3dadbabd2b91c32151/numpy/__init__.pyi#L4730-L4731
    """

    def __dlpack__(self, *, stream: None | _T_contra = ...) -> CapsuleType: ...

    def __dlpack_device__(self) -> tuple[int, Literal[0]]: ...


class DLDeviceType(enum.IntEnum):
    kDLCPU = (1,)
    kDLCUDA = (2,)
    kDLCUDAHost = (3,)
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
        x (Tensor): The input tensor, and the data type can be ``bool``, ``float16``, ``float32``,
            ``float64``, ``int8``, ``int16``, ``int32``, ``int64``, ``uint8``, ``complex64``,
            ``complex128``.

    Returns:
        dltensor, and the data type is PyCapsule.

    Examples:
        .. code-block:: python
            :name: code-paddle-to-paddle

            >>> import paddle
            >>> # x is a tensor with shape [2, 4]
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]])
            >>> dlpack = paddle.utils.dlpack.to_dlpack(x)
            >>> print(dlpack)
            >>> # doctest: +SKIP('the address will change in every run')
            <capsule object "dltensor" at 0x7f6103c681b0>
            >>> #doctest: -SKIP

            >>> # dlpack capsule will be renamed to 'used_dltensor' after decoded
            >>> y = paddle.utils.dlpack.from_dlpack(dlpack)
            >>> print(dlpack)
            >>> # doctest: +SKIP('the address will change in every run')
            <capsule object "used_dltensor" at 0x7f6103c681b0>

        .. code-block:: python
            :name: code-paddle-to-torch

            >>> # doctest: +SKIP('torch will not be installed')
            >>> # type: ignore
            >>> # convert tensor from paddle to other framework using to_dlpack
            >>> import torch

            >>> x = paddle.randn([2, 4]).to(device="cpu")
            >>> y = torch.from_dlpack(paddle.utils.dlpack.to_dlpack(x))
            >>> print(y.shape)
            torch.Size([2, 4])
            >>> # doctest: -SKIP
    """

    if in_dygraph_mode():
        if not isinstance(x, paddle.Tensor):
            raise TypeError(
                "The type of 'x' in to_dlpack must be paddle.Tensor,"
                f" but received {type(x)}."
            )

        return x.value().get_tensor()._to_dlpack()

    check_type(x, "x", (DenseTensor), "to_dlpack")
    return x._to_dlpack()


def from_dlpack(
    dlpack: SupportDLPack | CapsuleType,
) -> Tensor:
    """
    Decodes a DLPack to a tensor. The returned Paddle tensor will share the memory with
    the tensor from given dlpack.

    Args:
        dlpack (SupportDLPack | CapsuleType): A PyCapsule object with the dltensor,
            or that implements '__dlpack__' and '__dlpack_device__' methods.

            If `dlpack` is a tensor (or ndarray) object, it must support
            the `__dlpack__` protocol (i.e., have a `dlpack.__dlpack__`
            method). Otherwise `dlpack` may be a DLPack capsule, which is
            an opaque `PyCapsule` instance, typically produced by a
            `to_dlpack` function or method.

    Returns:
        out (Tensor): A tensor decoded from DLPack. The data type of returned tensor
            can be one of: ``int32``, ``int64``, ``float16``, ``float32`` and ``float64``.
            The device of returned tensor can be one of: ``CPU``, ``CUDAPlace``, ``CUDAPinnedPlace``.

    Examples:
        .. code-block:: python
            :name: code-paddle-from-paddle

            >>> import paddle
            >>> # From DLPack capsule
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]], place="cpu")
            >>> dlpack = paddle.utils.dlpack.to_dlpack(x)

            >>> y = paddle.utils.dlpack.from_dlpack(dlpack)
            >>> # dlpack capsule will be renamed to 'used_dltensor' after decoded
            >>> print(dlpack)
            >>> # doctest: +SKIP('the address will change in every run')
            <capsule object "used_dltensor" at 0x7f6103c681b0>

            >>> print(y)
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[0.20000000, 0.30000001, 0.50000000, 0.89999998],
                    [0.10000000, 0.20000000, 0.60000002, 0.69999999]])
            >>> # data of tensor x is shared with tensor y
            >>> y[0, 0] = 10.0
            >>> print(x)
            Tensor(shape=[2, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[10.       , 0.30000001, 0.50000000, 0.89999998],
                    [0.10000000, 0.20000000, 0.60000002, 0.69999999]])

        .. code-block:: python
            :name: code-paddle-from-numpy

            >>> # Directly from external tensor that implements '__dlpack__' and '__dlpack_device__' methods
            >>> import paddle
            >>> import numpy as np
            >>> x = np.array([[0.2, 0.3, 0.5, 0.9],
            ...              [0.1, 0.2, 0.6, 0.7]])
            >>> y = paddle.utils.dlpack.from_dlpack(x)
            >>> y[0, 0] = 10.0
            >>> # data of tensor x is shared with tensor y
            >>> print(x)
            [[10.   0.3  0.5  0.9]
            [ 0.1  0.2  0.6  0.7]]
    """

    if hasattr(dlpack, "__dlpack__"):
        device = dlpack.__dlpack_device__()
        # device is CUDA, we need to pass the current
        # stream
        if device[0] in (DLDeviceType.kDLCUDA,):
            with warnings.catch_warnings():
                # ignore deprecation warning
                warnings.filterwarnings("ignore", category=UserWarning)
                stream = paddle.device.cuda.current_stream(device[1])
            # cuda_stream is the pointer to the stream and it is a public
            # attribute, but it is not documented
            # The array API specify that the default legacy stream must be passed
            # with a value of 1 for CUDA
            # https://data-apis.org/array-api/latest/API_specification/array_object.html?dlpack-self-stream-none#dlpack-self-stream-none
            is_gpu = device[0] == DLDeviceType.kDLCUDA
            stream_ptr = (
                1 if is_gpu and stream.cuda_stream == 0 else stream.cuda_stream
            )
            dlpack_ = dlpack.__dlpack__(stream=stream_ptr)
        else:
            dlpack_ = dlpack.__dlpack__()
    else:
        # Old versions just call the converter
        dlpack_ = dlpack

    out: paddle.base.libpaddle.DenseTensor = paddle.base.core.from_dlpack(
        dlpack_
    )

    if in_dygraph_mode():
        out: Tensor = paddle.Tensor(out, place=out._place())

    return out
