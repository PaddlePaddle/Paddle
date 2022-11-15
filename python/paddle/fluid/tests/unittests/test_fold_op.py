#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()


class TestFoldOp(OpTest):
    """
    This is for test on fold Op
    """

    def init_data(self):
        self.batch_size = 3
        self.input_channels = 3 * 2 * 2
        self.length = 12
        self.kernel_sizes = [2, 2]
        self.strides = [1, 1]
        self.paddings = [0, 0, 0, 0]
        self.dilations = [1, 1]
        self.output_sizes = [4, 5]
        input_shape = [self.batch_size, self.input_channels, self.length]
        self.x = np.random.rand(*input_shape).astype(np.float64)

    def calc_fold(self):
        output_shape = [0] * 4
        output_shape[0] = self.batch_size
        output_shape[1] = int(
            self.input_channels / (self.kernel_sizes[0] * self.kernel_sizes[1])
        )
        output_shape[2] = self.output_sizes[0]
        output_shape[3] = self.output_sizes[1]
        dkernel_h = self.dilations[0] * (self.kernel_sizes[0] - 1) + 1
        dkernel_w = self.dilations[1] * (self.kernel_sizes[1] - 1) + 1
        col_height = (
            int(
                (
                    self.output_sizes[0]
                    + self.paddings[0]
                    + self.paddings[2]
                    - dkernel_h
                )
                / self.strides[0]
            )
            + 1
        )
        col_width = (
            int(
                (
                    self.output_sizes[1]
                    + self.paddings[1]
                    + self.paddings[3]
                    - dkernel_w
                )
                / self.strides[1]
            )
            + 1
        )
        output = np.zeros(output_shape).astype(np.float64)
        # ------------- calculate output ------------- #
        for b in range(output_shape[0]):
            for c in range(self.input_channels):
                w_offset = int(c % self.kernel_sizes[1])
                h_offset = int(
                    (c / self.kernel_sizes[1]) % self.kernel_sizes[0]
                )
                c_out = int(c / self.kernel_sizes[0] / self.kernel_sizes[1])
                for h in range(col_height):
                    h_out = int(
                        h * self.strides[0]
                        - self.paddings[0]
                        + h_offset * self.dilations[0]
                    )
                    for w in range(col_width):
                        w_out = int(
                            w * self.strides[1]
                            - self.paddings[1]
                            + w_offset * self.dilations[1]
                        )
                        if (h_out >= 0 and h_out < self.output_sizes[0]) and (
                            w_out >= 0 and w_out < self.output_sizes[1]
                        ):
                            output[b, c_out, h_out, w_out] += self.x[
                                b, c, w + col_width * h
                            ]

        self.outputs = output

    def set_data(self):
        self.init_data()
        self.calc_fold()
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.attrs = {
            'kernel_sizes': self.kernel_sizes,
            'paddings': self.paddings,
            'dilations': self.dilations,
            'strides': self.strides,
            'output_sizes': self.output_sizes,
        }
        self.outputs = {'Y': self.outputs}

    def setUp(self):
        self.op_type = 'fold'
        self.python_api = paddle.nn.functional.fold
        self.set_data()

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', check_eager=True)


class TestFoldAPI(TestFoldOp):

    # This is for test on paddle.nn.Fold

    def setUp(self):
        self.op_type = 'fold'
        self.python_api = paddle.nn.functional.fold
        self.set_data()
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_api(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input = paddle.to_tensor(self.x)
                m = paddle.nn.Fold(**self.attrs)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), self.outputs['Y'], rtol=1e-05
                )

    def test_info(self):
        str(paddle.nn.Fold(**self.attrs))


class TestFoldOpError(unittest.TestCase):
    def test_errors(self):
        from paddle.nn.functional import fold
        from paddle.fluid.framework import Program, program_guard

        with program_guard(Program(), Program()):

            def test_input_shape():
                # input_shpae must be 3-D
                x = paddle.randn(shape=[2, 3, 6, 7], dtype="float32")
                out = fold(x, output_sizes=[2, 3], kernel_sizes=[2, 2])

            def test_kernel_shape():
                # kernel_size must be 2
                x = paddle.randn(shape=[2, 6, 6], dtype="float32")
                out = fold(x, output_sizes=[2, 3], kernel_sizes=[2, 2, 3])

            def test_padding_shape():
                # padding_size must be 2 or 4
                x = paddle.randn(shape=[2, 6, 6], dtype="float32")
                out = fold(
                    x,
                    output_sizes=[2, 3],
                    kernel_sizes=[2, 2],
                    paddings=[2, 2, 3],
                )

            def test_dilations_shape():
                # dialtions_size must be 2
                x = paddle.randn(shape=[2, 6, 6], dtype="float32")
                out = fold(
                    x,
                    output_sizes=[2, 3],
                    kernel_sizes=[2, 2],
                    dilations=[2, 2, 3],
                )

            def test_strides_shape():
                # strids_size must be 2
                x = paddle.randn(shape=[2, 6, 6], dtype="float32")
                out = fold(
                    x,
                    output_sizes=[2, 3],
                    kernel_sizes=[2, 2],
                    strides=[2, 2, 3],
                )

            def test_output_size():
                # im_h * im_w must be L
                x = paddle.randn(shape=[2, 6, 6], dtype="float32")
                out = fold(
                    x, output_sizes=[6, 6], kernel_sizes=[2, 2], strides=[1, 1]
                )

            def test_output_size_2():
                # out_size must GT 1
                x = paddle.randn(shape=[2, 6, 6], dtype="float32")
                out = fold(
                    x,
                    output_sizes=[0.1, 0.2],
                    kernel_sizes=[2, 2],
                    strides=[1, 1],
                )

            def test_block_h_w():
                # test_block_h_w GT 0
                x = paddle.randn(shape=[2, 1, 1], dtype="float32")
                out = fold(
                    x, output_sizes=[1, 1], kernel_sizes=[2, 2], strides=1
                )

            def test_GT_0():
                x = paddle.randn(shape=[2, 1, 1], dtype="float32")
                out = fold(
                    x,
                    output_sizes=[0, 0],
                    kernel_sizes=[0, 0],
                    dilations=0,
                    paddings=[0, 0],
                    strides=0,
                )

            self.assertRaises(AssertionError, test_input_shape)
            self.assertRaises(AssertionError, test_kernel_shape)
            self.assertRaises(ValueError, test_padding_shape)
            self.assertRaises(AssertionError, test_dilations_shape)
            self.assertRaises(AssertionError, test_strides_shape)
            self.assertRaises(ValueError, test_output_size)
            self.assertRaises(TypeError, test_output_size_2)
            self.assertRaises(ValueError, test_block_h_w)
            self.assertRaises(ValueError, test_GT_0)


if __name__ == '__main__':
    unittest.main()
