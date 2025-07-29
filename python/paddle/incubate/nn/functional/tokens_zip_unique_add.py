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

from typing import TYPE_CHECKING

from paddle import _C_ops
from paddle.base.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor


def tokens_zip_unique_add(
    hidden_states_zipped: Tensor,
    hidden_states_unzipped: Tensor,
    idx_unzipped: Tensor,
    zipped_rows: int,
    name: str | None = None,
) -> tuple[Tensor]:
    r"""
    Partially unpermute tokens for Mixture of Experts (MoE) computation in distributed training scenarios.

    Args:
        hidden_states_zipped (Tensor): The input tensor containing tokens to be permuted, stored in row-major layout.
            Supported data types: bfloat16 or float8_e4m3fn.
            Shape: [sequence_length, token_dimension]
        hidden_states_unzipped (Tensor): The permuted and broadcasted input tensor.
            Shape: [total_tokens_after_broadcast, token_dimension]
            Data type: same as input hidden_states_zipped
        zipped_rows (int): Total rows of output y_zipped
        idx_unzipped(Tensor): Flattened expert indices aligned with given padding_alignment.
            Shape: [total_tokens_after_broadcast, 1]
            Data type: float32
        name (str|None, optional): Name prefix for the operation (optional).
            Default: None

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            - y_zipped (Tensor): Partially unpermuted tokens of given expert
                Shape: [zipped_rows, token_dimension]
                Data type: bfloat16
    """
    if in_dynamic_or_pir_mode():
        (y_zipped,) = _C_ops.tokens_zip_unique_add(
            hidden_states_zipped,
            hidden_states_unzipped,
            idx_unzipped,
            zipped_rows,
        )
        return y_zipped
