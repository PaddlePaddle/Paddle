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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestUniqueOp(OpTest):
    def setUp(self):
        self.op_type = "unique"
        self.init_config()

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64'), }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array(
                [2, 3, 1, 5], dtype='int64'),
            'Index': np.array(
                [0, 1, 1, 2, 3, 1], dtype='int32')
        }


class TestOne(TestUniqueOp):
    def init_config(self):
        self.inputs = {'X': np.array([2], dtype='int64'), }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array(
                [2], dtype='int64'),
            'Index': np.array(
                [0], dtype='int32')
        }


class TestRandom(TestUniqueOp):
    def init_config(self):
        self.inputs = {'X': np.random.randint(0, 100, (150, ), dtype='int64')}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(self.inputs['X'], True,
                                                       True)
        np_tuple = [(np_unique[i], np_index[i]) for i in range(len(np_unique))]
        np_tuple.sort(key=lambda x: x[1])
        target_out = np.array([i[0] for i in np_tuple], dtype='int64')
        target_index = np.array(
            [list(target_out).index(i) for i in self.inputs['X']],
            dtype='int64')

        self.outputs = {'Out': target_out, 'Index': target_index}


class TestUniqueRaiseError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            fluid.layers.unique([10])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data = fluid.data(shape=[10], dtype="float16", name="input")
            fluid.layers.unique(data)

        self.assertRaises(TypeError, test_dtype)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestOneGPU(TestUniqueOp):
    def init_config(self):
        self.inputs = {'X': np.array([2], dtype='int64'), }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array(
                [2], dtype='int64'),
            'Index': np.array(
                [0], dtype='int32')
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestRandomGPU(TestUniqueOp):
    def init_config(self):
        self.inputs = {'X': np.random.randint(0, 100, (150, ), dtype='int64')}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(self.inputs['X'], True,
                                                       True)
        np_tuple = [(np_unique[i], np_index[i]) for i in range(len(np_unique))]
        np_tuple.sort(key=lambda x: x[1])
        target_out = np.array([i[0] for i in np_tuple], dtype='int64')
        target_index = np.array(
            [list(target_out).index(i) for i in self.inputs['X']],
            dtype='int64')

        self.outputs = {'Out': target_out, 'Index': target_index}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
