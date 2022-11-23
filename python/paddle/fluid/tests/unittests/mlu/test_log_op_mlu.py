#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestActivation(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "exp"
        self.init_dtype()
        self.init_kernel_type()
        self.python_api = paddle.exp

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_dtype(self):
        self.dtype = np.float32

    def init_kernel_type(self):
        pass

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)
        __class__.no_need_check_grad = True


class TestLog(TestActivation):

    def setUp(self):
        self.set_mlu()
        self.op_type = "log"
        self.python_api = paddle.log
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_error(self):
        in1 = fluid.layers.data(name="in1",
                                shape=[11, 17],
                                append_batch_size=False,
                                dtype="int32")
        in2 = fluid.layers.data(name="in2",
                                shape=[11, 17],
                                append_batch_size=False,
                                dtype="int64")

        self.assertRaises(TypeError, fluid.layers.log, in1)
        self.assertRaises(TypeError, fluid.layers.log, in2)


class TestLog2(TestActivation):

    def setUp(self):
        self.set_mlu()
        self.op_type = "log2"
        self.python_api = paddle.log2
        self.init_dtype()

        x = np.random.uniform(1, 10, [11, 17]).astype(self.dtype)
        out = np.log2(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_error(self):
        in1 = paddle.static.data(name="in1", shape=[11, 17], dtype="int32")
        in2 = paddle.static.data(name="in2", shape=[11, 17], dtype="int64")

        self.assertRaises(TypeError, paddle.log2, in1)
        self.assertRaises(TypeError, paddle.log2, in2)

    def test_api(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float32")
            data_x = paddle.static.data(name="data_x",
                                        shape=[11, 17],
                                        dtype="float32")

            out1 = paddle.log2(data_x)
            exe = paddle.static.Executor(place=fluid.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            res1 = exe.run(paddle.static.default_main_program(),
                           feed={"data_x": input_x},
                           fetch_list=[out1])
        expected_res = np.log2(input_x)
        np.testing.assert_allclose(res1[0], expected_res, rtol=1e-6)

        # dygraph
        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float32")
            data_x = paddle.to_tensor(np_x)
            z = paddle.log2(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log2(np_x))
            np.savetxt("np_z.txt", np_z.flatten(), fmt="%.4f")
            np.savetxt("z_expected.txt", z_expected.flatten(), fmt="%.4f")
        np.testing.assert_allclose(np_z, z_expected, atol=1e-6)


class TestLog10(TestActivation):

    def setUp(self):
        self.set_mlu()
        self.op_type = "log10"
        self.python_api = paddle.log10
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log10(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_error(self):
        in1 = paddle.static.data(name="in1", shape=[11, 17], dtype="int32")
        in2 = paddle.static.data(name="in2", shape=[11, 17], dtype="int64")

        self.assertRaises(TypeError, paddle.log10, in1)
        self.assertRaises(TypeError, paddle.log10, in2)

    def test_api(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float32")
            data_x = paddle.static.data(name="data_x",
                                        shape=[11, 17],
                                        dtype="float32")

            out1 = paddle.log10(data_x)
            exe = paddle.static.Executor(place=paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            res1 = exe.run(paddle.static.default_main_program(),
                           feed={"data_x": input_x},
                           fetch_list=[out1])
        expected_res = np.log10(input_x)
        np.testing.assert_allclose(res1[0], expected_res, rtol=1e-6)

        # dygraph
        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float32")
            data_x = paddle.to_tensor(np_x)
            z = paddle.log10(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log10(np_x))
        np.testing.assert_allclose(np_z, z_expected, rtol=1e-4)


class TestLogHalf(TestLog):

    def init_dtype(self):
        self.dtype = np.float16

    def test_api(self):
        pass


class TestLog2Half(TestLog2):

    def init_dtype(self):
        self.dtype = np.float16

    def test_api(self):
        pass


class TestLog10Half(TestLog10):

    def init_dtype(self):
        self.dtype = np.float16

    def test_api(self):
        pass


if __name__ == '__main__':
    unittest.main()
