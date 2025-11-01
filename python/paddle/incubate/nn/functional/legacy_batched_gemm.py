
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, _C_ops
from paddle.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from collections.abc import Sequence


# special re-use of empty to reduce launch cost.
@functools.cache
def _empty_tensor() -> Tensor:
    """Get tensor with no entries and no data"""
    return Tensor()


def legacy_batched_gemm(
  lhs: Tensor, rhs: Tensor, batch_sizes: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """
    Cluster launched gemm into one op, which can be further fused and optimized.

    Args:
        lhs (Tensor): A tensor shaped in (total_seq_len, input_hidden_size), meant to be 
        perform gemm operation according to batch range.
        rhs (Tensor): A tensor shaped in (num_batches, input_hidden_size, output_hidden_size).
        batch_sizes(list): A list of integers representing the number of rows in each batch.

    Returns:
        tuple:
            - out (Tensor): The result of batched gemm operation.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.legacy_batched_gemm(lhs, rhs, batch_sizes)