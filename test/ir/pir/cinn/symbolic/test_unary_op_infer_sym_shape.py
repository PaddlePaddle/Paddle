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


class ArgMaxMinNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        argmax_out = paddle.argmax(x)
        argmin_out = paddle.argmin(x, axis=-1)
        return argmax_out, argmin_out


class TestArgMaxMinOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[0], data[NULL]',
                'shape[S0, S1], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = ArgMaxMinNet()

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
                net, input_spec, 'pd_op.argmax'
            )
            sym_shape_str_list += get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.argmin'
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


class AsComplexAsRealNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        real_res = paddle.as_complex(x)
        complex_res = paddle.as_real(real_res)
        return real_res, complex_res


class TestAsComplexAsRealOPInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[S0, S1], data[NULL]',
                'shape[S0, S1, 2], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = AsComplexAsRealNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(shape=[None, None, 2], dtype='float32')

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            sym_shape_str_list = get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.as_complex'
            )
            sym_shape_str_list += get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.as_real'
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


class CumSumProdNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        cumsum_out = paddle.cumsum(x)
        cumprod_out = paddle.cumprod(x, dim=1)
        return cumsum_out, cumprod_out


class TestCumSumProdOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[Mul(S0, S1, S2)], data[NULL]',
                'shape[S0, S1, S2], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = CumSumProdNet()

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
                net, input_spec, 'pd_op.cumsum'
            )
            sym_shape_str_list += get_sym_shape_str_for_op(
                net, input_spec, 'pd_op.cumprod'
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


class ReshapeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out1 = paddle.reshape(x, [-1, 4, 5])
        out2 = paddle.reshape(x, [0, 0, 12])
        return out1, out2


class TestReshapeOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[Mul(S0, S1, S2, 1 / (20)), 4, 5], data[NULL]',
                'shape[S0, S1, 12], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = ReshapeNet()

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
                net, input_spec, 'pd_op.reshape'
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


class DiagEmbedNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty([6])
        out = paddle.diag_embed(data)
        out = paddle.diag_embed(data, offset=-1, dim1=0, dim2=1)
        out = paddle.diag_embed(x)
        out = paddle.diag_embed(x, offset=-1, dim1=0, dim2=1)
        return out


class TestDiagEmbedOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[6, 6], data[NULL]',
                'shape[7, 7], data[NULL]',
                'shape[S0, S1, S2, S2], data[NULL]',
                'shape[Add(S2, 1), Add(S2, 1), S0, S1], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = DiagEmbedNet()

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
                net, input_spec, 'pd_op.diag_embed'
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


class DiagonalNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty([2, 2, 3], 'float32')
        out = paddle.diagonal(data)
        out = paddle.diagonal(data, offset=0, axis1=2, axis2=1)
        out = paddle.diagonal(x)
        out = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
        out = paddle.diagonal(x, offset=1, axis1=2, axis2=1)
        out = paddle.diagonal(x, offset=-1, axis1=2, axis2=1)
        return out


class TestDiagonalOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[3, Min(2, 2)], data[NULL]',
                'shape[2, Min(3, 2)], data[NULL]',
                'shape[S2, Min(S0, S1)], data[NULL]',
                'shape[S0, Min(S2, S1)], data[NULL]',
                'shape[S0, S3], data[NULL]',
                'shape[S0, S4], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = DiagonalNet()

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
                net, input_spec, 'pd_op.diagonal'
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


class KthvalueNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty([2, 3, 3], 'float32')
        out = paddle.kthvalue(data, 2, 1)
        return out


class TestKthvalueOpInferSymbolicShape(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[2, 3], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = KthvalueNet()

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
                net, input_spec, 'pd_op.kthvalue'
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
