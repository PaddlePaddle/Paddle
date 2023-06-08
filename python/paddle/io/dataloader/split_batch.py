#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numbers

import numpy as np

import paddle


def _split_field(field, micro_batch_size, acc_step):
    # duplicate
    if isinstance(field, (str, bytes, numbers.Number)):
        return (field for _ in range(acc_step))
    if isinstance(field, (np.ndarray, paddle.Tensor)):
        assert micro_batch_size * acc_step == field.shape[0]
        return (
            field[i * micro_batch_size : (i + 1) : micro_batch_size]
            for i in range(acc_step)
        )
    if isinstance(field, paddle.fluid.core.eager.Tensor):
        return (
            field[i * micro_batch_size : (i + 1) : micro_batch_size]
            for i in range(acc_step)
        )
    raise AssertionError("should not get here")


def _split_batch(flat_batch, micro_batch_size, acc_step):
    flat_micro_batches = (
        list(e)
        for e in (
            zip(
                [
                    _split_field(e, micro_batch_size, acc_step)
                    for e in flat_batch
                ]
            )
        )
    )
    return flat_micro_batches
