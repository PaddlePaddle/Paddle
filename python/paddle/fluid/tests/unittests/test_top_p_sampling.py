#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from eager_op_test import OpTest

import paddle
from paddle.fluid import core


def TopPProcess(probs, top_p):
    sorted_probs = paddle.sort(probs, descending=True)
    sorted_indices = paddle.argsort(probs, descending=True)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep the first token
    sorted_indices_to_remove = paddle.cast(
        sorted_indices_to_remove, dtype='int64'
    )
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # Scatter sorted tensors to original indexing
    sorted_indices = (
        sorted_indices
        + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
    )
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(),
        sorted_indices.flatten(),
        sorted_indices_to_remove.flatten(),
    )
    condition = paddle.cast(condition, 'bool').reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    next_tokens = paddle.multinomial(probs)
    next_scores = paddle.index_sample(probs, next_tokens)
    return next_scores, next_tokens


class TestTopPSamplingOp(OpTest):
    def init_args(self):
        self.topp = 0.0
        self.batch_size = 1
        self.vocab_size = 10000
        self.seed = 2023
        self.dtype = "float32"

    def _get_places(self):
        places = []
        places.append(core.CUDAPlace(0))
        return places

    def setUp(self):
        self.init_args()
        self.op_type = "top_p_sampling"
        self.python_api = paddle.top_p_sampling
        self.public_python_api = paddle.top_p_sampling
        self.input_data = np.random.rand(self.batch_size, self.vocab_size)
        self.topp_tensor = paddle.to_tensor(
            [
                self.topp,
            ]
            * self.batch_size,
            self.dtype,
        ).reshape((-1, 1))
        self.inputs = {
            'x': paddle.to_tensor(self.input_data, self.dtype),
            'ps': self.topp_tensor,
        }
        self.attrs = {'seed': self.seed}
        next_scores, next_tokens = TopPProcess(
            paddle.to_tensor(self.input_data, self.dtype), self.topp
        )
        self.outputs = {'out': next_scores, 'ids': next_tokens}

    def test_check_output(self):
        self.check_output()


class TestTopPSamplingOp1(TestTopPSamplingOp):
    def init_args(self):
        self.topp = 0.0
        self.batch_size = 10
        self.vocab_size = 100000
        self.seed = 2023
        self.dtype = "float16"


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
