# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

if TYPE_CHECKING:
    import paddle


def _check_data_shape_to_num_outputs(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_outputs: int,
    allow_1d_reshape: bool = False,
) -> None:
    """Check that predictions and target have the correct shape, else raise error.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting
        allow_1d_reshape: Allow that for num_outputs=1 that preds and target does not need to be 1d tensors. Instead
            code that follows are expected to reshape the tensors to 1d.

    """
    if preds.ndim > 2 or target.ndim > 2:
        raise ValueError(
            f"Expected both predictions and target to be either 1- or 2-dimensional tensors, but got {target.ndim} and {preds.ndim}."
        )
    cond1 = False
    if not allow_1d_reshape:
        cond1 = num_outputs == 1 and not (
            preds.ndim == 1 or preds.shape[1] == 1
        )
    cond2 = num_outputs > 1 and preds.ndim > 1 and num_outputs != preds.shape[1]
    if cond1 or cond2:
        raise ValueError(
            f"Expected argument `num_outputs` to match the second dimension of input, but got {num_outputs} and {preds.shape[1]}."
        )
