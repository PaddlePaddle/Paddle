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

import sys
import unittest
from os.path import dirname

import numpy as np
from test_infer_sym_shape_utils import (
    TestBase,
    check_infer_results,
)

import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
from utils import apply_to_static


class ExpandNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = paddle.expand(x, [paddle.shape(y)[1], paddle.shape(y)[0]])
        out = paddle.expand(x, [7, 5, paddle.shape(y)[0]])
        out = paddle.expand(x, [7, -1, paddle.shape(y)[0]])
        out = paddle.expand(x, [7, paddle.shape(y)[1], -1])

        return out


class ExpandOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.x = paddle.rand([1, 3], 'float32')
        self.y = paddle.rand([3, 2], 'float32')
        self.expected = [
            'shape[S3, S2], data[NULL]',
            'shape[7, 5, S2], data[NULL]',
            'shape[7, S0, S2], data[NULL]',
            'shape[7, S3, S1], data[NULL]',
        ]

    @unittest.skip("TODO: xiongkun")
    def test_eval_symbolic(self):
        net = ExpandNet()
        input_spec = [
            InputSpec(shape=[None, None], dtype='float32'),
            InputSpec(shape=[None, None], dtype='float32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.expand', self.expected)
        out = net(self.x, self.y)
        return out


class MeshgridNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out_x, out_y = paddle.meshgrid(x, y)
        return out_x, out_y


class MeshgridOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.x_cases = [
            np.random.rand(1),
            np.random.rand(10),
            np.random.rand(100),
            np.random.rand(1000),
        ]
        self.y_cases = [
            np.random.rand(1),
            np.random.rand(10),
            np.random.rand(1000),
            np.random.rand(100),
        ]

        self.expected = [
            'shape[S0, S1], data[NULL], shape[S0, S1], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = MeshgridNet()

        for i in range(len(self.x_cases)):
            x = self.x_cases[i]
            y = self.y_cases[i]
            x_spec = InputSpec(
                shape=[None for _ in range(len(x.shape))], dtype='float32'
            )
            y_spec = InputSpec(
                shape=[None for _ in range(len(y.shape))], dtype='float32'
            )

            input_spec = [x_spec, y_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            paddle.core._set_prim_forward_blacklist("pd_op.meshgrid")
            check_infer_results(
                net, input_spec, 'pd_op.meshgrid', self.expected
            )

        # TODO(WintersMontagne10335): Add builtin.meshgrid op infer symbolic shape test
        #                Not added because attribute `sym_shape_str` does not support multi-output op now.
        #                See also: paddle/fluid/pir/transforms/shape_optimization_pass.cc:144.


class LinspaceNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.linspace(start=0, stop=5, num=10)
        return out


class LinspaceOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[10], data[NULL]']

    def test_eval_symbolic(self):
        net = LinspaceNet()
        x_spec = InputSpec(shape=[None, None, 2], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.linspace', self.expected)
        return True


class LogspaceNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.logspace(start=1, stop=5, num=10)
        return out


class LogspaceOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[10], data[NULL]']

    def test_eval_symbolic(self):
        net = LogspaceNet()
        x_spec = InputSpec(shape=[None, None, 2], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.logspace', self.expected)
        return True


class SliceNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = x[:, -1, :]
        return out


class SliceOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = ['shape[S0, S2], data[NULL]']

    @unittest.skip("TODO: xiongkun")
    def test_eval_symbolic(self):
        net = SliceNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(net, input_spec, 'pd_op.slice', self.expected)

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


class TakeAlongAxisOpInferSymbolicShapeTest(TestBase):
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

    @unittest.skip("TODO: xiongkun")
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
            check_infer_results(
                net, input_spec, 'pd_op.take_along_axis', self.expected[i]
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


class TransposeOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 3, 4)]

        self.expected = [
            'shape[S1, S0, S2], data[NULL]',
            'shape[4], data[2, 3, 2, 2]',
        ]

    @unittest.skip("TODO: xiongkun")
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
            check_infer_results(
                net, input_spec, 'pd_op.transpose', self.expected
            )

        return True


class PoissonNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.poisson(x)

        return out


class PoissonOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 3, 4)]
        self.expected = ['shape[S0, S1, S2], data[NULL]']

    def test_eval_symbolic(self):
        net = PoissonNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(net, input_spec, 'pd_op.poisson', self.expected)

        return True


class TrilNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.tril(x)

        return out


class TrilOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 3, 4)]
        self.expected = ['shape[S0, S1, S2], data[NULL]']

    @unittest.skip("TODO: xiongkun")
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
            check_infer_results(net, input_spec, 'pd_op.tril', self.expected)

        return True


class InterpolateNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        input_data = paddle.empty(shape=(2, 3, 6, 10))
        output = F.interpolate(x=input_data, size=[12, 12])
        return output


class InterpolateOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.x = paddle.rand([1, 3], 'float32')
        self.expected = [
            'shape[2, 3, 12, 12], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = InterpolateNet()
        input_spec = [
            InputSpec(shape=[None, None], dtype='float32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'pd_op.nearest_interp', self.expected
        )
        out = net(self.x)
        return out


if __name__ == '__main__':
    unittest.main()
