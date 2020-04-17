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
import paddle.fluid as fluid
from inference_pass_test import InferencePassTest


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
