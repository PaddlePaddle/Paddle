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

import paddle
from paddle.static import InputSpec


def get_sym_shape_str_for_op(net, input_spec, op_name='builtin.shadow_output'):
    forward_program = net.forward.get_concrete_program(*input_spec)[
        1
    ].infer_program.forward_program
    all_sym_shape_str = []
    for op in forward_program.global_block().ops:
        if op.name() == op_name:
            all_sym_shape_str.append(op.attrs()['sym_shape_str'])

    return all_sym_shape_str


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


class TestBase(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        pass

    def test_eval_symbolic(self):
        pass


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


class TestEmbeddingOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.x_shape = [1, 2048]
        self.num_embeddings = 32000
        self.embedding_dim = 768
        self.x = paddle.randint(low=0, high=768, shape=self.x_shape)
        self.expected_sym_shape = 'shape[S0, S1, 768], data[NULL]'

    def test_eval_symbolic(self):
        net = EmbeddingNet(self.num_embeddings, self.embedding_dim)
        input_spec = [
            InputSpec(shape=[None, None], dtype='float32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()

        # check the infer result
        sym_shape_str_list = get_sym_shape_str_for_op(
            net, input_spec, 'builtin.shadow_output'
        )
        np.testing.assert_equal(len(sym_shape_str_list), 1)
        np.testing.assert_equal(
            sym_shape_str_list[0].find(self.expected_sym_shape),
            0,
            'output shape is not expected!',
        )
        out = net(self.x)
        return out


class ArangeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, in_0, in_1, in_2):
        if in_1 is None:
            end = in_0
            out = paddle.arange(end)
        else:
            start, end, step = in_0, in_1, in_2
            out = paddle.arange(start, end, step)

        return out


class TestArangeOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.start = paddle.full([1], 0)
        self.end = paddle.full([1], 5)
        self.step = paddle.full([1], 1)

        self.expected_sym_shape = (
            'shape[Mul(Add(S1, -S0), 1 / (S2))], data[NULL]'
        )

    def test_eval_symbolic(self):
        net = ArangeNet()

        input_spec = [
            InputSpec(shape=[None], dtype='float32'),
            InputSpec(shape=[None], dtype='float32'),
            InputSpec(shape=[None], dtype='float32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()

        # check the infer result
        sym_shape_str_list = get_sym_shape_str_for_op(net, input_spec)
        np.testing.assert_equal(len(sym_shape_str_list), 1)
        np.testing.assert_equal(
            sym_shape_str_list[0].find(self.expected_sym_shape),
            0,
            'output shape is not expected!',
        )
        out = net(self.start, self.end, self.step)
        return out


class ExpandNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = paddle.expand(x, [paddle.shape(y)[1], paddle.shape(y)[0]])
        out = paddle.expand(x, [7, 5, paddle.shape(y)[0]])
        out = paddle.expand(x, [7, -1, paddle.shape(y)[0]])
        out = paddle.expand(x, [7, paddle.shape(y)[1], -1])

        return out


class TestExpandOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.x = paddle.rand([1, 3], 'float32')
        self.y = paddle.rand([3, 2], 'float32')

        self.expected_sym_shapes = [
            'shape[S3, S2], data[NULL]',
            'shape[7, 5, S2], data[NULL]',
            'shape[7, S0, S2], data[NULL]',
            'shape[7, S3, S1], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = ExpandNet()

        input_spec = [
            InputSpec(shape=[None, None], dtype='float32'),
            InputSpec(shape=[None, None], dtype='float32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()

        # check the infer result
        sym_shape_str_list = get_sym_shape_str_for_op(
            net, input_spec, 'pd_op.expand'
        )
        np.testing.assert_equal(
            len(sym_shape_str_list), len(self.expected_sym_shapes)
        )
        for i in range(len(self.expected_sym_shapes)):
            np.testing.assert_equal(
                sym_shape_str_list[i].find(self.expected_sym_shapes[i]),
                0,
                'output shape is not expected!',
            )
        out = net(self.x, self.y)
        return out


class MatmulNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, trans_x, trans_y):
        out = paddle.matmul(x, y, trans_x, trans_y)

        return out


class TestMatmulOpInferSymbolicShape(TestBase):
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

            # check the infer result
            sym_shape_str_list = get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.matmul'
            )
            np.testing.assert_equal(len(sym_shape_str_list), 1)
            np.testing.assert_equal(
                sym_shape_str_list[0].find(self.expected[i]),
                0,
                f'in case i = {i}: output shape ({sym_shape_str_list[0]}) is not expected {(self.expected[i])}',
            )

        return True


class MaxNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.max(x)
        out = paddle.max(x, 0)
        out = paddle.max(x, 1)
        out = paddle.max(x, -1)
        out = paddle.max(x, -2)

        # keepdim=True
        # out = paddle.max(x, 0, True)

        return out


class TestMaxOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 4)]

        self.expected = [
            [
                'shape[], data[NULL]',
                'shape[S1], data[NULL]',
                'shape[S0], data[NULL]',
                'shape[S0], data[NULL]',
                'shape[S1], data[NULL]',
                # 'shape[1, S1], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = MaxNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            sym_shape_str_list = get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.max'
            )
            np.testing.assert_equal(
                len(sym_shape_str_list), len(self.expected[i])
            )
            for j in range(len(sym_shape_str_list)):
                np.testing.assert_equal(
                    sym_shape_str_list[j].find(self.expected[i][j]),
                    0,
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[0]}) is not expected {(self.expected[i][j])}',
                )

        return True


if __name__ == '__main__':
    unittest.main()
