# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.base import core


def _print_tensor_in_gpu(tensor):
    """
    Print a GPU tensor's dtype, shape, and all data values directly from
    the device using a single-thread CUDA kernel (device-side printf).

    This function is **CUDA Graph safe**: no host/device memory transfer
    is performed (shape is passed via kernel-argument registers), so it
    can be called inside a CUDA Graph capture region.

    Args:
        tensor (paddle.Tensor): A GPU DenseTensor to print. Must already
            reside on a CUDA device (call ``tensor.cuda()`` first if needed).

    Raises:
        ValueError: If PaddlePaddle is not compiled with CUDA support.
        InvalidArgument: If the tensor is not a DenseTensor or not on GPU.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> paddle.utils._print_tensor_in_gpu(x)

    """
    if not core.is_compiled_with_cuda():
        raise ValueError(
            "paddle.utils._print_tensor_in_gpu is not supported in "
            "CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU "
            "support to call this API."
        )
    core.eager._print_tensor_in_gpu(tensor)
