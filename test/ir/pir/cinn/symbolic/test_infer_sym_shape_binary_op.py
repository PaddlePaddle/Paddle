# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from test_infer_sym_shape_utils import (
    TestBase,
    apply_to_static,
    check_infer_results,
)

import paddle
from paddle.static import InputSpec


class EmbeddingNet(paddle.nn.Layer):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = paddle.nn.Embedding(
            num_embeddings,
            embedding_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal()
            ),
        )

    def forward(self, x):
        out = self.embedding(x)
        return out


class EmbeddingOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.x_shape = [1, 2048]
        self.num_embeddings = 32000
        self.embedding_dim = 768
        self.x = paddle.randint(low=0, high=768, shape=self.x_shape)
        self.expected = ['shape[S0, S1, 768], data[NULL]']

    def test_eval_symbolic(self):
        net = EmbeddingNet(self.num_embeddings, self.embedding_dim)
        input_spec = [
            InputSpec(shape=[None, None], dtype='float32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'builtin.shadow_output', self.expected
        )
        out = net(self.x)
        return out


class KronNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = paddle.empty(shape=[2, 2])
        z = paddle.empty(shape=[3, 3])
        out = paddle.kron(x, y)
        out = paddle.kron(y, z)
        return out


class KronOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = [
            'shape[S0, Mul(S1, 2), Mul(S2, 2)], data[NULL]',
            'shape[6, 6], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = KronNet()

        x_spec = InputSpec(shape=[None, None, None], dtype='float32')

        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.kron', self.expected)

        return True


class MatmulNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, trans_x, trans_y):
        out = paddle.matmul(x, y, trans_x, trans_y)
        return out


class MatmulOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [
            # [x, y, trans_x, trans_y]
            [np.random.rand(1, 3), np.random.rand(3, 2), False, False],
            # with broadcast
            [np.random.rand(10), np.random.rand(10), False, False],  # []
            [np.random.rand(10, 5), np.random.rand(5), False, False],  # [10]
            [
                np.random.rand(10, 5, 2),
                np.random.rand(2),
                False,
                False,
            ],  # [10, 5]
            [
                np.random.rand(10, 5, 2),
                np.random.rand(10, 2, 5),
                False,
                False,
            ],  # [10, 5, 5]
            [
                np.random.rand(10, 1, 5, 2),
                np.random.rand(1, 3, 2, 5),
                False,
                False,
            ],  # [10, 3, 5, 5]
            # with transpose
            [np.random.rand(3, 5), np.random.rand(3, 2), True, False],  # [5, 2]
            [np.random.rand(3, 5), np.random.rand(4, 5), False, True],  # [3, 4]
        ]

        self.expected = [
            'shape[S0, S3], data[NULL]',
            # with broadcast
            'shape[], data[NULL]',
            'shape[S0], data[NULL]',
            'shape[S0, S1], data[NULL]',
            'shape[Broadcast(S0, S3), S1, S5], data[NULL]',
            'shape[Broadcast(S0, S4), Broadcast(S1, S5), S2, S7], data[NULL]',
            # with transpose
            'shape[S1, S3], data[NULL]',
            'shape[S0, S2], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = MatmulNet()

        for i in range(len(self.cases)):
            x, y, trans_x, trans_y = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )
            y_spec = InputSpec(
                shape=[None for index in range(len(y.shape))], dtype='float32'
            )

            input_spec = [x_spec, y_spec, trans_x, trans_y]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            expected_symbol = [self.expected[i]]
            check_infer_results(
                net, input_spec, 'pd_op.matmul', expected_symbol
            )

        return True


class Conv2dNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv2D(4, 6, (3, 3))

    def forward(self, x):
        z = paddle.empty(shape=[2, 4, 8, 8])
        out = self.conv(z)
        return out


class Conv2dOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[2, 6, 6, 6], data[NULL]']

    def test_eval_symbolic(self):
        net = Conv2dNet()

        x_spec = InputSpec(shape=[None, None, None], dtype='float32')

        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.conv2d', self.expected)

        return True


if __name__ == '__main__':
    unittest.main()
