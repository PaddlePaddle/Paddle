# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle.framework import core

paddle.enable_static()


def manual_embeddingbag(input, params, weights=None, mode="sum"):
    bag = params[input]
    if weights is not None:
        bag *= np.expand_dims(weights, 2)
    if mode == "sum":
        return np.sum(bag, axis=-2)
    else:
        assert mode == "mean"
        assert weights is None
        return np.mean(bag, axis=-2)


# each row of input should avoid repeated elements
def get_input(rows=5, cols=3, num_embeddings=10):
    a = np.random.choice(np.arange(num_embeddings), size=cols, replace=False)
    for _ in range(rows - 1):
        b = np.random.choice(np.arange(num_embeddings), size=cols, replace=False)
        a = np.vstack((a, b))
    return a


class TestEmbeddingBagCPU(OpTest):
    def setUp(self):
        self.op_type = "embedding_bag"
        self.dtype = "float64"
        self.ids_dtype = "int64"
        self.mode = "sum"
        self.python_api = paddle.nn.functional.embedding_bag
        weight = np.random.random((20, 64)).astype(self.dtype)
        input = get_input(10, 20, weight.shape[0])
        per_sample_weight = np.random.randint(low=0, high=10, size=input.shape).astype(
            np.float64
        )

        self.inputs = {'input': input, 'weight': weight, 'per_sample_weight': per_sample_weight}
        np_out = manual_embeddingbag(input, weight, per_sample_weight)
        self.outputs = {
            'out': np_out.reshape((input.shape[0], weight.shape[1]))
        }
        self.attrs = {'mode': self.mode}
        if core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            inputs_to_check=['per_sample_weight'],
            output_names=['out'],
            max_relative_error=0.5,
        )

    def test_check_grad_weight(self):
        self.check_grad(
            inputs_to_check=['weight'],
            output_names=['out'],
            max_relative_error=0.5,
        )


if __name__ == '__main__':
    unittest.main()
