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
from op_test import OpTest, _set_use_system_allocator
import paddle
import paddle.fluid as fluid


paddle.enable_static()

@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGatherOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "gather"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Index': OpTest.np_dtype_to_fluid_dtype(self.index)
        }
        self.attrs = {'validate_indices': 'true'}
        self.outputs = {'Out': self.out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_input_output(self):
        self.x = np.array([[1, 2], [3, 4], [5, 6]]).astype(self.dtype)
        self.index = np.array([1, 2]).astype(self.dtype)
        self.out = np.array([[3, 4], [5, 6]]).astype(self.dtype)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGatherAPI(unittest.TestCase):
    def test_name(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 2], dtype="float32")
            index = paddle.static.data(name='index', shape=[1], dtype='float32')

            out = paddle.gather(x, index, name='gather')
            self.assertEqual(('gather' in out.name), True)

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):

            x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype('float32')
            index_np = np.array([1, 2]).astype('float32')

            x = paddle.static.data(name="x", shape=[3, 2], dtype='float32')
            index = paddle.static.data(name="index", shape=[1], dtype='float32')

            z = paddle.gather(x, index)

            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            x_value, index_value, z_value = exe.run(
                feed={"x": x_np,
                      "index": index_np}, fetch_list=[x, index, z])

            z_expected = np.array([[3, 4], [5, 6]])
            self.assertEqual(
                (x_value == x_np).all(),
                True,
                msg="x_value = {}, but expected {}".format(x_value, x_np))
            self.assertEqual(
                (index_value == index_np).all(),
                True,
                msg="index_value = {}, but expected {}".format(index_value,
                                                               index_np))
            self.assertEqual(
                (z_value == z_expected).all(),
                True,
                msg="z_value = {}, but expected {}".format(z_value, z_expected))

    def test_backward(self):
        # TODO(ascendrc): Test backward after add grad npu op implemented.
        pass


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGatherError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # the input of gather must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.NPUPlace(0))
            index1 = fluid.create_lod_tensor(
                np.array([-1, 3]), [[1, 1, 1, 1]], fluid.NPUPlace(0))
            self.assertRaises(TypeError, paddle.gather, x1, index1)

            # the input dtype must be float16 or float32 or float64 or int32 or int64
            x2 = paddle.static.data(
                name='x2', shape=[3, 4, 5, 6], dtype="uint8")
            index2 = paddle.static.data(
                name='index2', shape=[3, 4, 5, 6], dtype="uint8")
            self.assertRaises(TypeError, paddle.gather, x2, index2)


if __name__ == '__main__':
    unittest.main()
