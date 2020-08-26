#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
# import paddle.fluid.layers as layers
# import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_


class TestEmptyOp(OpTest):
    def setUp(self):
        self.op_type = "empty"
        self.init_config()

    def test_check_output(self):
        # self.check_output()
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        # print('----- outs -----: ', outs)
        total_value = np.sum(np.array(outs[0]))
        mean_value = np.mean(np.array(outs[0]))
        # print('total_value: ', total_value)
        # print('mean_value: ', mean_value)
        always_non_zero = total_value != 0.0 and mean_value != 0.0
        self.assertTrue(always_non_zero, 'always non zeros.')
        # always_zero = total_value == 0.0 and mean_value == 0.0
        # self.assertTrue(always_zero or always_non_zero,
        #                 'always_zero or always_non_zero.')

    def init_config(self):
        shape = [500, 3]
        dtype = 'float32'
        # dtype = core.VarDesc.VarType.FP32
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        # print('----- dtype_inner -----: ', dtype_inner)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyOp2(TestEmptyOp):
    def init_config(self):
        shape = [500, 3]
        dtype = 'float64'
        # dtype = core.VarDesc.VarType.FP64
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        # print('----- dtype_inner -----: ', dtype_inner)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyOp3(TestEmptyOp):
    def init_config(self):
        shape = [500, 3]
        dtype = 'int32'
        # dtype = core.VarDesc.VarType.INT32
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        # print('----- dtype_inner -----: ', dtype_inner)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyOp4(TestEmptyOp):
    def init_config(self):
        shape = [500, 3]
        dtype = 'int64'
        # dtype = core.VarDesc.VarType.INT64
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        # print('----- dtype_inner -----: ', dtype_inner)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyAPI(unittest.TestCase):
    def __check_out__(self, out, dtype='float32'):
        # print('-----out-----:', out)
        total_value = np.sum(np.array(out), dtype=dtype)
        mean_value = np.mean(np.array(out), dtype=dtype)
        # print('total_value: ', total_value)
        # print('mean_value: ', mean_value)
        always_non_zero = total_value != 0.0 and mean_value != 0.0
        # self.assertTrue(always_non_zero, 'always non zeros.')
        always_zero = total_value == 0.0 and mean_value == 0.0
        self.assertTrue(always_zero or always_non_zero,
                        'always_zero or always_non_zero.')

    def test_dygraph_api_out(self):
        paddle.disable_static()
        shape = [200, 3]
        out = paddle.empty(shape=shape)
        self.__check_out__(out)
        paddle.enable_static()

    def test_dygraph_api_attr(self):
        paddle.disable_static()
        shape = [200, 3]
        dtype = 'float64'
        out = paddle.empty(shape=shape, dtype=dtype)
        self.__check_out__(out, dtype)
        paddle.enable_static()

    def test_static_graph(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            shape = [200, 3]
            dtype = 'float64'
            out = paddle.empty(shape=shape, dtype=dtype)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result = exe.run(feed={}, fetch_list=[out])
        self.__check_out__(result[0][0], dtype)


class TestEmptyError(unittest.TestCase):
    def test_attr(self):
        def test_dtype():
            shape = [200, 3]
            dtype = 'bool'
            result = paddle.empty(shape=shape, dtype=dtype)

        self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    unittest.main()
