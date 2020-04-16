# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
from ir.inference.inference_pass_test import InferencePassTest

import paddle.fluid as fluid


@skip_check_grad_ci(reason="Tests inference only optimization.")
class TestMatMulOpSpecial(OpTest):
    def generate_data(self):
        self.x = np.random.random([1, 128, 128]).astype("float32")
        self.y = np.random.random([1, 128, 64]).astype("float32")
        self.out = np.matmul(self.x, self.y)
        self.shape_out = []
        self.axis_out = []

    def setUp(self):
        self.op_type = "matmul"
        self._cpu_only = True
        self.use_mkldnn = True
        self.generate_data()

        self.inputs = {'X': self.x, 'Y': self.y}
        self.attrs = {'use_mkldnn': self.use_mkldnn, }
        if len(self.shape_out) > 0:
            self.attrs['reshape_Out'] = self.shape_out
            self.attrs['axis_Out'] = self.axis_out

        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(reason="Tests inference only optimization.")
class TestMatMulOpSpecialSimplest(TestMatMulOpSpecial):
    def generate_data(self):
        bs = 3
        self.x = np.random.random([bs, 12, 128, 128]).astype("float32")
        self.y = np.random.random([bs, 12, 128, 64]).astype("float32")
        self.axis_out = [0, 2, 1, 3]
        self.shape_out = [0, 0, 768]
        self.out = np.matmul(self.x, self.y).transpose([0, 2, 1, 3]).reshape(
            [bs, -1, 768])


class TestMKLDNNMatmulFuseOp(InferencePassTest):
    def setUp(self):
        bs = 8
        d_type = 'float32'
        shape_x = [12, 128, 128]
        shape_y = [12, 128, 64]
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(name='x', shape=[-1] + shape_x, dtype=d_type)
            y = fluid.data(name='y', shape=[-1] + shape_y, dtype=d_type)
            out = fluid.layers.matmul(x, y)
            out = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
            out = fluid.layers.reshape(out, [0, 0, shape_y[0] * shape_y[2]])
            out = fluid.layers.fc(out, size=1)

        self.feeds = {
            "x": np.random.random([bs] + shape_x).astype(d_type),
            "y": np.random.random([bs] + shape_y).astype(d_type)
        }
        self.fetch_list = [out]
        self.enable_mkldnn = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)


class TestMKLDNNMatmulOtherDimsFuseOp(InferencePassTest):
    def setUp(self):
        bs = 8
        d_type = 'float32'
        shape_x = [12, 1, 1]
        shape_y = [12, 1, 64]
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(name='x', shape=[-1] + shape_x, dtype=d_type)
            y = fluid.data(name='y', shape=[-1] + shape_y, dtype=d_type)
            out = fluid.layers.matmul(x, y)
            out = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
            out = fluid.layers.reshape(out, [0, 0, shape_y[0] * shape_y[2]])
            out = fluid.layers.fc(out, size=1)

        self.feeds = {
            "x": np.random.random([bs] + shape_x).astype(d_type),
            "y": np.random.random([bs] + shape_y).astype(d_type)
        }
        self.fetch_list = [out]
        self.enable_mkldnn = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)


class TestMKLDNNMatmulNotFuseOp(InferencePassTest):
    def setUp(self):
        batch_size = 7
        d_type = 'float32'
        shape_x = [12, 128, 128]
        shape_y = [12, 128, 64]
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(name='x', shape=[-1] + shape_x, dtype=d_type)
            y = fluid.data(name='y', shape=[-1] + shape_y, dtype=d_type)
            out = fluid.layers.matmul(x, y)
            out = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
            out = fluid.layers.transpose(
                out, perm=[0, 1, 2, 3])  # breaks pattern
            out = fluid.layers.reshape(out, [0, 0, 768])
            out = fluid.layers.fc(out, size=1)

        self.feeds = {
            "x": np.random.random([batch_size] + shape_x).astype(d_type),
            "y": np.random.random([batch_size] + shape_y).astype(d_type)
        }
        self.fetch_list = [out]
        self.enable_mkldnn = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)


if __name__ == '__main__':
    unittest.main()
