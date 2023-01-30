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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestUniqueWithCountsOp(OpTest):
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestUniqueWithCountsOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "unique_with_counts"
        self.init_config()

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.inputs = {
            'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64'),
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array([2, 3, 1, 5], dtype='int64'),
            'Index': np.array([0, 1, 1, 2, 3, 1], dtype='int32'),
<<<<<<< HEAD
            'Count': np.array([1, 3, 1, 1], dtype='int32'),
=======
            'Count': np.array([1, 3, 1, 1], dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestOne(TestUniqueWithCountsOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_config(self):
        self.inputs = {
            'X': np.array([2], dtype='int64'),
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array([2], dtype='int64'),
            'Index': np.array([0], dtype='int32'),
<<<<<<< HEAD
            'Count': np.array([1], dtype='int32'),
=======
            'Count': np.array([1], dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestRandom(TestUniqueWithCountsOp):
<<<<<<< HEAD
    def init_config(self):
        input_data = np.random.randint(0, 100, (2000,), dtype='int64')
        self.inputs = {'X': input_data}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(
            self.inputs['X'], True, True
        )
=======

    def init_config(self):
        input_data = np.random.randint(0, 100, (2000, ), dtype='int64')
        self.inputs = {'X': input_data}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(self.inputs['X'], True,
                                                       True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np_tuple = [(np_unique[i], np_index[i]) for i in range(len(np_unique))]
        np_tuple.sort(key=lambda x: x[1])
        target_out = np.array([i[0] for i in np_tuple], dtype='int64')
        target_index = np.array(
<<<<<<< HEAD
            [list(target_out).index(i) for i in self.inputs['X']], dtype='int64'
        )
=======
            [list(target_out).index(i) for i in self.inputs['X']],
            dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        count = [0 for i in range(len(np_unique))]
        for i in range(target_index.shape[0]):
            count[target_index[i]] += 1
        target_count = np.array(count, dtype='int64')
        self.outputs = {
            'Out': target_out,
            'Index': target_index,
<<<<<<< HEAD
            'Count': target_count,
=======
            'Count': target_count
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestUniqueWithCountsRaiseError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        def test_type():
            paddle.unique([10])
=======

    def test_errors(self):

        def test_type():
            fluid.layers.unique_with_counts([10])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data = fluid.data(shape=[10], dtype="float16", name="input")
<<<<<<< HEAD
            paddle.unique(data)
=======
            fluid.layers.unique_with_counts(data)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(TypeError, test_dtype)


<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestOneGPU(TestUniqueWithCountsOp):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestOneGPU(TestUniqueWithCountsOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_config(self):
        self.inputs = {
            'X': np.array([2], dtype='int64'),
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
            'Out': np.array([2], dtype='int64'),
            'Index': np.array([0], dtype='int32'),
<<<<<<< HEAD
            'Count': np.array([1], dtype='int32'),
=======
            'Count': np.array([1], dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)


<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestRandomGPU(TestUniqueWithCountsOp):
    def init_config(self):
        input_data = np.random.randint(0, 100, (2000,), dtype='int64')
        self.inputs = {'X': input_data}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(
            self.inputs['X'], True, True
        )
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestRandomGPU(TestUniqueWithCountsOp):

    def init_config(self):
        input_data = np.random.randint(0, 100, (2000, ), dtype='int64')
        self.inputs = {'X': input_data}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT64)}
        np_unique, np_index, reverse_index = np.unique(self.inputs['X'], True,
                                                       True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np_tuple = [(np_unique[i], np_index[i]) for i in range(len(np_unique))]
        np_tuple.sort(key=lambda x: x[1])
        target_out = np.array([i[0] for i in np_tuple], dtype='int64')
        target_index = np.array(
<<<<<<< HEAD
            [list(target_out).index(i) for i in self.inputs['X']], dtype='int64'
        )
=======
            [list(target_out).index(i) for i in self.inputs['X']],
            dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        count = [0 for i in range(len(np_unique))]
        for i in range(target_index.shape[0]):
            count[target_index[i]] += 1
        target_count = np.array(count, dtype='int64')
        self.outputs = {
            'Out': target_out,
            'Index': target_index,
<<<<<<< HEAD
            'Count': target_count,
=======
            'Count': target_count
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
