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

import unittest
import numpy as np

import paddle.fluid as fluid
from inference_pass_test import InferencePassTest


class TestMKLDNNMatmulFuseOp(InferencePassTest):

    def init_data(self):
        self.bs = 8
        self.d_type = np.float32
        self.shape_x = [12, 128, 128]
        self.shape_y = [12, 128, 64]
        self.enable_mkldnn = True

    def make_network(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(name='x',
                           shape=[-1] + self.shape_x,
                           dtype=self.d_type)
            y = fluid.data(name='y',
                           shape=[-1] + self.shape_y,
                           dtype=self.d_type)
            out = fluid.layers.matmul(x, y)
            out = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
            out = fluid.layers.reshape(
                out, [0, 0, self.shape_y[0] * self.shape_y[2]])
            out = fluid.layers.fc(out, size=1)
        return out

    def setUp(self):
        self.init_data()
        out = self.make_network()
        self.set_feeds(out)

    def set_feeds(self, out):
        self.feeds = {
            "x": np.random.random([self.bs] + self.shape_x).astype(self.d_type),
            "y": np.random.random([self.bs] + self.shape_y).astype(self.d_type)
        }
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)


class TestMKLDNNMatmulOtherDimsFuseOp(TestMKLDNNMatmulFuseOp):

    def init_data(self):
        self.bs = 8
        self.d_type = np.float32
        self.shape_x = [12, 1, 1]
        self.shape_y = [12, 1, 64]
        self.enable_mkldnn = True


class TestMKLDNNMatmulOpNotFusedWrongTransposeAxis(TestMKLDNNMatmulFuseOp):

    def make_network(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(name='x',
                           shape=[-1] + self.shape_x,
                           dtype=self.d_type)
            y = fluid.data(name='y',
                           shape=[-1] + self.shape_y,
                           dtype=self.d_type)
            out = fluid.layers.matmul(x, y)
            out = fluid.layers.transpose(out, perm=[0, 1, 2, 3])
            out = fluid.layers.reshape(out, [0, 0, 0, 0])
            out = fluid.layers.fc(out, size=1)
        return out


class TestMKLDNNMatmulOpNotFusedBreakPattern(TestMKLDNNMatmulFuseOp):

    def init_data(self):
        self.bs = 7
        self.d_type = np.float32
        self.shape_x = [12, 128, 128]
        self.shape_y = [12, 128, 64]
        self.enable_mkldnn = True

    def make_network(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(name='x',
                           shape=[-1] + self.shape_x,
                           dtype=self.d_type)
            y = fluid.data(name='y',
                           shape=[-1] + self.shape_y,
                           dtype=self.d_type)
            out = fluid.layers.matmul(x, y)
            out = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
            out = fluid.layers.transpose(out, perm=[0, 1, 2,
                                                    3])  # breaks pattern
            out = fluid.layers.reshape(
                out, [0, 0, self.shape_y[0] * self.shape_y[2]])
            out = fluid.layers.fc(out, size=1)
        return out


if __name__ == '__main__':
    unittest.main()
