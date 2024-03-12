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
        out = paddle.max(x, keepdim=True)
        out = paddle.max(x, 0, keepdim=True)
        out = paddle.max(x, 1, keepdim=True)
        out = paddle.max(x, -1, keepdim=True)
        out = paddle.max(x, -2, keepdim=True)

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
                # keepdim=True
                'shape[1, 1], data[NULL]',
                'shape[1, S1], data[NULL]',
                'shape[S0, 1], data[NULL]',
                'shape[S0, 1], data[NULL]',
                'shape[1, S1], data[NULL]',
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
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[j]}) is not expected {(self.expected[i][j])}',
                )

        return True


class TakeAlongAxisNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices):
        out = paddle.take_along_axis(x, indices, axis=0)
        out = paddle.take_along_axis(x, indices, axis=1)
        out = paddle.take_along_axis(x, indices, axis=-1)
        out = paddle.take_along_axis(x, indices, axis=-2)

        return out


class TestTakeAlongAxisOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [
            [
                np.random.rand(2, 3, 4),
                np.ones([6, 3, 4], dtype='int32'),
            ],
        ]

        self.expected = [
            [
                'shape[S3, S1, S2], data[NULL]',
                'shape[S0, S4, S2], data[NULL]',
                'shape[S0, S1, S5], data[NULL]',
                'shape[S0, S4, S2], data[NULL]',
            ],
        ]

    def test_eval_symbolic(self):
        net = TakeAlongAxisNet()

        for i in range(len(self.cases)):
            x, indices = self.cases[i]
            x_spec = InputSpec(
                shape=[None for _ in range(len(x.shape))], dtype='float32'
            )
            indices_spec = InputSpec(
                shape=[None for _ in range(len(indices.shape))], dtype='int32'
            )

            input_spec = [x_spec, indices_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            sym_shape_str_list = get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.take_along_axis'
            )
            np.testing.assert_equal(
                len(sym_shape_str_list), len(self.expected[i])
            )
            for j in range(len(sym_shape_str_list)):
                np.testing.assert_equal(
                    sym_shape_str_list[j].find(self.expected[i][j]),
                    0,
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[j]}) is not expected {(self.expected[i][j])}',
                )

        return True


class PutAlongAxisNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices, value):
        out = paddle.put_along_axis(x, indices, value, axis=0)
        out = paddle.put_along_axis(x, indices, value, axis=1)
        out = paddle.put_along_axis(x, indices, value, axis=-1)

        return out


class TestPutAlongAxisOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [
            [
                np.random.rand(2, 3, 4),
                np.ones([2, 3, 4], dtype='int32'),
                np.ones([2, 3, 4], dtype='float32'),
            ],
        ]

        self.expected = [
            [
                'shape[S0, S1, S2], data[NULL]',
                'shape[S0, S1, S2], data[NULL]',
                'shape[S0, S1, S2], data[NULL]',
            ],
        ]

    def test_eval_symbolic(self):
        net = PutAlongAxisNet()

        for i in range(len(self.cases)):
            x, indices, value = self.cases[i]
            x_spec = InputSpec(
                shape=[None for _ in range(len(x.shape))], dtype='float32'
            )
            indices_spec = InputSpec(
                shape=[None for _ in range(len(indices.shape))], dtype='int32'
            )
            value_spec = InputSpec(
                shape=[None for _ in range(len(value.shape))], dtype='float32'
            )

            input_spec = [x_spec, indices_spec, value_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            sym_shape_str_list = get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.put_along_axis'
            )
            np.testing.assert_equal(
                len(sym_shape_str_list), len(self.expected[i])
            )
            for j in range(len(sym_shape_str_list)):
                np.testing.assert_equal(
                    sym_shape_str_list[j].find(self.expected[i][j]),
                    0,
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[j]}) is not expected {(self.expected[i][j])}',
                )

        return True


class TransposeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.transpose(x, perm=[1, 0, 2])

        x = x.reshape([2, 3, 2, 2])
        shape = paddle.shape(x)
        out = shape.transpose(perm=(0,))

        return out


class TestTransposeOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 3, 4)]

        self.expected = [
            ['shape[S1, S0, S2], data[NULL]', 'shape[4], data[2, 3, 2, 2]']
        ]

    def test_eval_symbolic(self):
        net = TransposeNet()

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
                net, input_spec, 'pd_op.transpose'
            )
            np.testing.assert_equal(
                len(sym_shape_str_list), len(self.expected[i])
            )
            for j in range(len(sym_shape_str_list)):
                np.testing.assert_equal(
                    sym_shape_str_list[j].find(self.expected[i][j]),
                    0,
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[j]}) is not expected {(self.expected[i][j])}',
                )

        return True


class TrilNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.tril(x)

        return out


class TestTrilOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 3, 4)]

        self.expected = [
            [
                'shape[S0, S1, S2], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = TrilNet()

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
                net, input_spec, 'pd_op.tril'
            )
            np.testing.assert_equal(
                len(sym_shape_str_list), len(self.expected[i])
            )
            for j in range(len(sym_shape_str_list)):
                np.testing.assert_equal(
                    sym_shape_str_list[j].find(self.expected[i][j]),
                    0,
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[j]}) is not expected {(self.expected[i][j])}',
                )

        return True


class SliceNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = x[:, -1, :]
        # out = x[1:3, 0:2, 2:4]

        # axes = [0, 1, 2]
        # starts = [-3, 0, 2]
        # ends = [3, 2, 4]
        # out = paddle.slice(x, axes=axes, starts=starts, ends=ends)

        return out


class TestSliceOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]

        self.expected = [
            [
                'shape[S0, S2], data[NULL]',
                # 'shape[2, 2, 2], data[NULL]',
                # 'shape[Add(3, -Add(-3, S0)), 2, 2]',
            ]
        ]

    def test_eval_symbolic(self):
        net = SliceNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            # net = apply_to_static(net, False, input_spec)
            net = apply_to_static(net, True, input_spec)
            net.eval()

            # check the infer result
            sym_shape_str_list = get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.slice'
            )
            np.testing.assert_equal(
                len(sym_shape_str_list), len(self.expected[i])
            )
            for j in range(len(sym_shape_str_list)):
                np.testing.assert_equal(
                    sym_shape_str_list[j].find(self.expected[i][j]),
                    0,
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[j]}) is not expected {(self.expected[i][j])}',
                )

        return True


class SplitNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.split(x, [-1], axis=1)
        out = paddle.split(x, [1, 2, -1], axis=1)
        out = paddle.split(x, [1, -1], axis=1)
        out = paddle.split(x, [1, 2, 3], axis=1)
        out = paddle.split(x, [1, 2, x.shape[1]], axis=1)

        out = x.split([-1], axis=1)
        out = x.split([1, 2, -1], axis=1)
        out = x.split([1, -1], axis=1)
        out = x.split([1, 2, 3], axis=1)
        out = x.split([1, 2, x.shape[1]], axis=1)

        return out


class TestSplitOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 6, 5)]

        self.expected = [
            [
                'shape[S0, S1, S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, 2, S2], data[NULL], shape[S0, Add(S1, -3), S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, Add(S1, -1), S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, 2, S2], data[NULL], shape[S0, 3, S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, 2, S2], data[NULL], shape[S0, S1, S2], data[NULL]',
                'shape[S0, S1, S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, 2, S2], data[NULL], shape[S0, Add(S1, -3), S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, Add(S1, -1), S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, 2, S2], data[NULL], shape[S0, 3, S2], data[NULL]',
                'shape[S0, 1, S2], data[NULL], shape[S0, 2, S2], data[NULL], shape[S0, S1, S2], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = SplitNet()

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
                net, input_spec, 'pd_op.split'
            )
            np.testing.assert_equal(
                len(sym_shape_str_list), len(self.expected[i])
            )
            for j in range(len(sym_shape_str_list)):
                np.testing.assert_equal(
                    sym_shape_str_list[j].find(self.expected[i][j]),
                    0,
                    f'in case i,j = {i},{j}: output shape ({sym_shape_str_list[j]}) is not expected {(self.expected[i][j])}',
                )

        # TODO(fty1777): Add builtin.split op infer symbolic shape test
        #                Not added because attribute `sym_shape_str` does not support multi-output op now.
        #                See also: paddle/fluid/pir/transforms/shape_optimization_pass.cc:144.

        return True


if __name__ == '__main__':
    unittest.main()
