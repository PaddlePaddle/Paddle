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
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.framework import Operator
from paddle.fluid.tests.unittests.mkldnn.test_fc_mkldnn_op import fully_connected_naive
from paddle.fluid.tests.unittests.mkldnn.test_conv2d_mkldnn_op import conv2d_bias_naive

fluid.core.globals()['FLAGS_tensor_dump_ops'] = 'fc_mkldnn_fwd,conv_mkldnn_fwd'


def load_array_from_file(filename, num):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f]
        assert len(lines) >= num * 6
        num_count = 0
        results = [[0] * 6 for i in range(num)]
        out = [None] * num
        while (num_count < num):
            for j in range(6):
                results[num_count][j] = lines[num_count * 6 + j].split(": ")[-1]
            num_count = num_count + 1

            for i in range(num):
                assert eval(results[i][0]) == i
                out[i] = [eval(k) for k in results[i][5][1:-1].split(" ")]
            return out


class TestTensorDump(unittest.TestCase):
    def test_check_output(self):
        self._cpu_only = True
        self.conv_x = np.random.random((2, 3, 5, 5)).astype("float32")
        self.conv_w = np.random.random((250, 3, 5, 5)).astype("float32")
        self.conv_bias = np.random.random((250)).astype("float32")
        self.conv_out = np.random.random((2, 250)).astype("float32")
        self.w = np.random.random((10 * 5 * 5, 18)).astype("float32")
        self.bias = np.random.random(18).astype("float32")
        self.fc_out = fully_connected_naive(self.conv_out, self.w, self.bias)

        place = fluid.core.CPUPlace()
        var_dict = {
            'conv_x': self.conv_x,
            'conv_w': self.conv_w,
            'conv_bias': self.conv_bias,
            'conv_out': self.conv_out,
            'w': self.w,
            'bias': self.bias,
            'fc_out': self.fc_out
        }

        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in var_dict:
                block.create_var(
                    name=name, dtype=np.float32, shape=var_dict[name].shape)

            block.append_op(
                type="conv2d",
                inputs={
                    'Input': block.var('conv_x'),
                    'Filter': block.var('conv_w'),
                    'Bias': block.var('conv_bias')
                },
                outputs={'Output': block.var('conv_out')},
                attrs={'use_mkldnn': True})

            block.append_op(
                type="fc",
                inputs={
                    'Input': block.var('conv_out'),
                    'W': block.var('w'),
                    'Bias': block.var('bias')
                },
                outputs={'Out': block.var('fc_out')},
                attrs={'use_mkldnn': True})

        exe = fluid.Executor(place)
        conv_out, fc_out = exe.run(program,
                                   feed={
                                       'conv_x': self.conv_x,
                                       'conv_w': self.conv_w,
                                       'conv_bias': self.conv_bias,
                                       'w': self.w,
                                       'bias': self.bias
                                   },
                                   fetch_list=['conv_out', 'fc_out'])

        conv_out_in_file = load_array_from_file(
            "out/conv_mkldnn_fwd_conv_out_float_ANY_LAYOUT", 1)
        fc_out_in_file = load_array_from_file(
            "out/fc_mkldnn_fwd_fc_out_float_ANY_LAYOUT", 1)
        np.allclose(
            np.array(conv_out_in_file[0]), conv_out.flatten(), atol=1e-4)
        np.allclose(np.array(fc_out_in_file[0]), fc_out.flatten(), atol=1e-4)
