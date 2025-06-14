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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import paddle
from paddle import Tensor

_num_sms = None


def set_num_sms(num_sms: int) -> None:
    """
    Set the maximum SM count for all GEMM kernels to use.

    Arguments:
        num_sms: the desired maximum SM count for all GEMM kernels to use.
    """
    global _num_sms
    assert (
        0
        < num_sms
        <= paddle.device.cuda.get_device_properties(
            device="cuda"
        ).multi_processor_count
    )
    _num_sms = num_sms


def get_num_sms() -> int:
    """
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    """
    global _num_sms
    if _num_sms is None:
        _num_sms = (
            paddle.device.cuda.get_device_properties().multi_processor_count
        )
    return _num_sms


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def get_m_alignment_for_contiguous_layout():
    """
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.

    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    """
    return 128


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


def get_col_major_tma_aligned_tensor(x: Tensor) -> Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `paddle.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    if x.dim() == 2:
        m, n = x.shape

        aligned_m = get_tma_aligned_size(m, x.element_size())

        if aligned_m == m and x.strides[0] == 1 and x.strides[1] == aligned_m:
            return x

        x, remove_dim = x.unsqueeze(0), True

    b, m, n = x.shape
    aligned_m = get_tma_aligned_size(m, x.element_size())

    # The last kernel gives a column-major TMA aligned layout
    if (
        x.strides[0] == aligned_m * n
        and x.strides[1] == 1
        and x.strides[2] == aligned_m
    ):
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = paddle.transpose(
        paddle.empty((b, n, aligned_m), dtype=x.dtype), perm=[0, 2, 1]
    )
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x
