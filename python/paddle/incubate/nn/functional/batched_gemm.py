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

from paddle import Tensor, _C_ops
from paddle.framework import in_dynamic_or_pir_mode


def batched_gemm(
    lhs: Tensor,
    rhs: Tensor,
    batch_sizes: list,
    trans_lhs: bool = False,
    trans_rhs: bool = False,
) -> tuple[Tensor]:
    """
    Cluster launched gemm into one op, which can be further fused and optimized.

    Args:
        lhs (Tensor): A tensor shaped in (total_seq_len, input_hidden_size), meant to be
        perform gemm operation according to batch range.
        rhs (Tensor): A tensor shaped in (num_batches, input_hidden_size, output_hidden_size).
        batch_sizes(list): A list of integers representing the number of rows in each batch.
        trans_lhs (bool): Whether view lhs matrix as last 2D-transposed. Default: False.
        trans_rhs (bool): Whether view rhs matrix as last 2D-transposed. Default: False.

    Returns:
        tuple:
            - out (Tensor): The result of batched gemm operation.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.batched_gemm(lhs, rhs, batch_sizes, trans_lhs, trans_rhs)
