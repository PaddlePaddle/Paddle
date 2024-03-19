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


class ArgMaxMinNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        argmax_out = paddle.argmax(x)
        argmin_out = paddle.argmin(x, axis=-1)
        return argmax_out, argmin_out


class ArgMaxMinOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            ['shape[0], data[NULL]'],
            ['shape[S0, S1], data[NULL]'],
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
            check_infer_results(
                net, input_spec, 'pd_op.argmax', self.expected[0]
            )
            check_infer_results(
                net, input_spec, 'pd_op.argmin', self.expected[1]
            )

        return True


class AsComplexAsRealNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        complex_res = paddle.as_complex(x)
        real_res = paddle.as_real(complex_res)
        return real_res, complex_res


class AsComplexAsRealOPInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            ['shape[S0, S1], data[NULL]'],
            ['shape[S0, S1, 2], data[NULL]'],
        ]

    def test_eval_symbolic(self):
        net = AsComplexAsRealNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(shape=[None, None, 2], dtype='float32')

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'pd_op.as_complex', self.expected[0]
            )
            check_infer_results(
                net, input_spec, 'pd_op.as_real', self.expected[1]
            )

        return True


class CumSumProdNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        cumsum_out = paddle.cumsum(x)
        cumprod_out = paddle.cumprod(x, dim=1)
        return cumsum_out, cumprod_out


class CumSumProdOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            ['shape[Mul(S0, S1, S2)], data[NULL]'],
            ['shape[S0, S1, S2], data[NULL]'],
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
            check_infer_results(
                net, input_spec, 'pd_op.cumsum', self.expected[0]
            )
            check_infer_results(
                net, input_spec, 'pd_op.cumprod', self.expected[1]
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


class DiagEmbedOpInferSymbolicShapeTest(TestBase):
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
            check_infer_results(
                net, input_spec, 'pd_op.diag_embed', self.expected[0]
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


class DiagonalOpInferSymbolicShapeTest(TestBase):
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

            check_infer_results(
                net, input_spec, 'pd_op.diagonal', self.expected[0]
            )

        return True


class KthvalueNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty([2, 3, 3], 'float32')
        out = paddle.kthvalue(data, 2, 1)
        return out


class KthvalueOpInferSymbolicShapeTest(TestBase):
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
            check_infer_results(
                net, input_spec, 'pd_op.kthvalue', self.expected[0]
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


class MaxOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 4)]

        self.expected = [
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
            check_infer_results(net, input_spec, 'pd_op.max', self.expected)

        return True


class PutAlongAxisNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices, value):
        out = paddle.put_along_axis(x, indices, value, axis=0)
        out = paddle.put_along_axis(x, indices, value, axis=1)
        out = paddle.put_along_axis(x, indices, value, axis=-1)

        return out


class PutAlongAxisOpInferSymbolicShapeTest(TestBase):
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
            check_infer_results(
                net, input_spec, 'pd_op.put_along_axis', self.expected[i]
            )

        return True


class ReshapeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out1 = paddle.reshape(x, [-1, 4, 5])
        out2 = paddle.reshape(x, [0, 0, 12])
        return out1, out2


class ReshapeOpInferSymbolicShapeTest(TestBase):
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

            check_infer_results(
                net, input_spec, 'pd_op.reshape', self.expected[0]
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


class SplitOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 6, 5)]
        self.expected = [
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
            check_infer_results(net, input_spec, 'pd_op.split', self.expected)

        # TODO(fty1777): Add builtin.split op infer symbolic shape test
        #                Not added because attribute `sym_shape_str` does not support multi-output op now.
        #                See also: paddle/fluid/pir/transforms/shape_optimization_pass.cc:144.
        return True


if __name__ == '__main__':
    unittest.main()
