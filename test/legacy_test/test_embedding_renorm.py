#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.nn.functional.input import embedding_renorm_


def ref_embedding_renorm_(x, weight, max_norm, norm_type=2.0):
    weight = weight.copy()
    x = np.reshape(x, (-1,))
    x = np.unique(x)
    x = np.sort(x)
    for i in range(len(x)):
        norm = np.linalg.norm(
            weight[int(x[i])], ord=norm_type, axis=0, keepdims=False
        )
        if norm > max_norm:
            weight[int(x[i])] *= max_norm / (norm + 1e-7)
    return weight


class TestEmbeddingRenormOp(unittest.TestCase):
    def setUp(self):
        self._init_attr()
        self.dtype = self._init_dtype()
        x = np.array([[2, 1, 3], [4, 5, 6]]).astype("int64")
        weight = np.random.random((10, 4)).astype(self.dtype) * 10
        y_ref = ref_embedding_renorm_(x, weight, self.max_norm, self.norm_type)
        self.inputs = {'X': x, 'Weight': weight}
        self.outputs = {'Out': y_ref}
        self.attrs = {'max_norm': self.max_norm, 'norm_type': self.norm_type}

    def _init_dtype(self):
        return "float32"

    def _init_attr(self):
        self.max_norm = 1.0
        self.norm_type = 2.0

    def test_check_output(self):
        paddle_result = embedding_renorm_(
            paddle.to_tensor(self.inputs['X']),
            paddle.to_tensor(self.inputs['Weight']),
            self.max_norm,
            self.norm_type,
        )
        np.testing.assert_allclose(
            paddle_result.numpy(), self.outputs['Out'], atol=1e-5
        )


class TestEmbeddingRenormOp1(TestEmbeddingRenormOp):
    def _init_attr(self):
        self.max_norm = 1.0
        self.norm_type = 1.0


class TestEmbeddingRenormOp2(TestEmbeddingRenormOp):
    def _init_attr(self):
        self.max_norm = 1.0
        self.norm_type = 3.0


if __name__ == '__main__':
    unittest.main()
