#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base

paddle.enable_static()


class XPUTestUnfoldOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'unfold'
        self.use_dynamic_create_class = False

    class TestUnfoldOp(XPUOpTest):
        """
        This is for test on unfold Op
        """

        def init_data(self):
            self.batch_size = 3
            self.input_channels = 3
            self.input_height = 20
            self.input_width = 20

            self.kernel_sizes = [2, 2]
            self.strides = [1, 1]
            self.paddings = [1, 1, 1, 1]
            self.dilations = [1, 1]
            input_shape = [
                self.batch_size,
                self.input_channels,
                self.input_height,
                self.input_width,
            ]
            self.x = np.random.rand(*input_shape).astype(self.dtype)

        def calc_unfold(self):
            output_shape = [0] * 3
            output_shape[0] = self.batch_size
            output_shape[1] = (
                self.input_channels
                * self.kernel_sizes[0]
                * self.kernel_sizes[1]
            )
            dkernel_h = self.dilations[0] * (self.kernel_sizes[0] - 1) + 1
            dkernel_w = self.dilations[1] * (self.kernel_sizes[1] - 1) + 1
            out_height = (
                int(
                    (
                        self.input_height
                        + self.paddings[0]
                        + self.paddings[2]
                        - dkernel_h
                    )
                    / self.strides[0]
                )
                + 1
            )
            out_width = (
                int(
                    (
                        self.input_width
                        + self.paddings[1]
                        + self.paddings[3]
                        - dkernel_w
                    )
                    / self.strides[1]
                )
                + 1
            )
            output_shape[2] = out_height * out_width
            output = np.zeros(output_shape).astype(np.float64)
            # ------------ calculate output -------------- #
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    for k in range(output_shape[2]):
                        h_out = int(k / out_width)
                        w_out = k % out_width
                        w_offset = j % self.kernel_sizes[1]
                        h_offset = (
                            int(j / self.kernel_sizes[1]) % self.kernel_sizes[0]
                        )
                        c_in = int(
                            j / (self.kernel_sizes[0] * self.kernel_sizes[1])
                        )
                        h_in = (
                            h_offset * self.dilations[0]
                            + h_out * self.strides[0]
                            - self.paddings[0]
                        )
                        w_in = (
                            w_offset * self.dilations[1]
                            + w_out * self.strides[1]
                            - self.paddings[1]
                        )
                        if (h_in >= 0 and h_in < self.input_height) and (
                            w_in >= 0 and w_in < self.input_width
                        ):
                            output[i, j, k] = self.x[i, c_in, h_in, w_in]

            self.outputs = output

        def set_data(self):
            self.init_data()
            self.calc_unfold()

            self.inputs = {'X': self.x}
            self.attrs = {
                'kernel_sizes': self.kernel_sizes,
                'paddings': self.paddings,
                'dilations': self.dilations,
                'strides': self.strides,
            }
            self.outputs = {'Y': self.outputs}

        def setUp(self):
            self.op_type = 'unfold'
            self.dtype = self.in_type
            self.set_data()

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Y')

    class TestUnfoldAPI(TestUnfoldOp):
        """
        This is for test on paddle.nn.Unfold
        """

        def setUp(self):
            self.op_type = 'unfold'
            self.set_data()
            self.places = [paddle.XPUPlace(0)]

        def test_dygraph(self):
            for place in self.places:
                with base.dygraph.guard(place):
                    input = paddle.to_tensor(self.inputs['X'])
                    m = paddle.nn.Unfold(**self.attrs)
                    m.eval()
                    result = m(input)
                    np.testing.assert_allclose(
                        result.numpy(), self.outputs['Y'], rtol=1e-05
                    )

        def test_info(self):
            str(paddle.nn.Unfold(**self.attrs))


support_types = get_xpu_op_support_types('unfold')
for stype in support_types:
    create_test_class(globals(), XPUTestUnfoldOp, stype)

if __name__ == "__main__":
    unittest.main()
