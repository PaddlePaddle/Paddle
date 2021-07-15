#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestEqual(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = x == y  # all elements are not equal

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLessthan(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "less_than"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = x < y

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGreaterthan(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "greater_than"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x, y = self.get_input()
        out = self.get_output(x, y)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        return x, y

    def get_output(self, x, y):
        return x > y

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNotequal(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "not_equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x, y = self.get_input()
        out = self.get_output(x, y)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        return x, y

    def get_output(self, x, y):
        return x != y

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLessequal(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "less_equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x, y = self.get_input()
        out = self.get_output(x, y)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        return x, y

    def get_output(self, x, y):
        return x <= y

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestEqual2(TestEqual):
    def setUp(self):
        self.set_npu()
        self.op_type = "equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        out = x == y  # all elements are equal, except position [0][1]

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}


class TestLessthan2(TestLessthan):
    def setUp(self):
        self.set_npu()
        self.op_type = "less_than"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        out = x < y  # all elements are equal, except position [0][1]

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}


class TestGreaterthan2(TestGreaterthan):
    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        return x, y


class TestGreaterthanBroadcast0(TestGreaterthan):
    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.array([1.5]).astype(self.dtype)
        return x, y


class TestGreaterthanBroadcast1(TestGreaterthan):
    def get_input(self):
        x = np.random.uniform(1, 2, [1, 17]).astype(self.dtype)
        y = x.T
        return x, y


class TestGreaterthanBroadcast2(TestGreaterthan):
    def get_input(self):
        x = np.random.uniform(1, 2, [2, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [2, 1, 17]).astype(self.dtype)
        return x, y


class TestGreaterthanBroadcast3(TestGreaterthan):
    def get_input(self):
        x = np.random.uniform(1, 2, [1, 2, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [2, 1, 17]).astype(self.dtype)
        return x, y


class TestNotequal2(TestNotequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        return x, y


class TestNotequalBroadcast0(TestNotequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.array([1.5]).astype(self.dtype)
        return x, y


class TestNotequalBroadcast1(TestNotequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [1, 17]).astype(self.dtype)
        y = x.T
        return x, y


class TestNotequalBroadcast2(TestNotequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [2, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [2, 1, 17]).astype(self.dtype)
        return x, y


class TestNotequalBroadcast3(TestNotequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [1, 2, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [2, 1, 17]).astype(self.dtype)
        return x, y


class TestLessequal2(TestLessequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        return x, y


class TestLessequalBroadcast0(TestLessequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.array([1.5]).astype(self.dtype)
        return x, y


class TestLessequalBroadcast1(TestLessequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [1, 17]).astype(self.dtype)
        y = x.T
        return x, y


class TestLessequalBroadcast2(TestLessequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [2, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [2, 1, 17]).astype(self.dtype)
        return x, y


class TestLessequalBroadcast3(TestLessequal):
    def get_input(self):
        x = np.random.uniform(1, 2, [1, 2, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [2, 1, 17]).astype(self.dtype)
        return x, y


class TestEqual2FP16(TestEqual2):
    def init_dtype(self):
        self.dtype = np.float16


class TestEqual2Int(TestEqual2):
    def init_dtype(self):
        self.dtype = np.int32


class TestLessthan2FP16(TestLessthan2):
    def init_dtype(self):
        self.dtype = np.float16


class TestGreaterthan2FP16(TestGreaterthan2):
    def init_dtype(self):
        self.dtype = np.float16


class TestGreaterthan2Int(TestGreaterthan2):
    def init_dtype(self):
        self.dtype = np.int32


class TestNotequal2FP16(TestNotequal2):
    def init_dtype(self):
        self.dtype = np.float16


class TestNotequal2Int(TestNotequal2):
    def init_dtype(self):
        self.dtype = np.int32


class TestLessequal2FP16(TestLessequal2):
    def init_dtype(self):
        self.dtype = np.float16


class TestLessequal2Int(TestLessequal2):
    def init_dtype(self):
        self.dtype = np.int32


if __name__ == '__main__':
    unittest.main()
