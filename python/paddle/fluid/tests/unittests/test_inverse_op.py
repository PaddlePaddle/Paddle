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
import paddle.fluid.core as core
import paddle
from op_test import OpTest


class TestInverseOp(OpTest):
    def config(self):
        self.matrix_shape = [100, 100]

    def setUp(self):
        self.op_type = "inverse"
        self.config()

        mat = np.random.random(self.matrix_shape).astype("float64")
        inverse = np.linalg.inv(mat)

        self.inputs = {'Input': mat}
        self.outputs = {'Output': inverse}

    def test_check_output(self):
        self.check_output()

    def test_grad(self):
        self.check_grad(['Input'], 'Output')


class TestInverseOpBatched(TestInverseOp):
    def config(self):
        self.matrix_shape = [4, 32, 32]


class TestInverseAPI(unittest.TestCase):
    def check_static_result(self, place, N, with_out=False):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[N, N], dtype="float32")
            if with_out:
                out = fluid.data(name="output", shape=[N, N], dtype="float32")
            else:
                out = None
            result = paddle.inverse(input=input, out=None)

            input_np = np.random.random([N, N]).astype("float32")
            result_np = np.linalg.inv(input_np)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(
                np.allclose(
                    fetches[0], result_np, atol=1e-4),
                "Output has diff at " + str(place) + " when the shape is " +
                str(input.shape) + ". The maximum diff is " +
                str(np.amax(np.absolute(fetches[0] - result_np))))

    def test_static(self):
        for N in [4, 36]:
            self.check_static_result(place=fluid.CPUPlace(), N=N)
            if core.is_compiled_with_cuda():
                self.check_static_result(place=fluid.CUDAPlace(0), N=N)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            input_np = np.array([[2, 0], [0, 2]]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = paddle.inverse(input)
            self.assertTrue(
                np.allclose(result.numpy(), np.linalg.inv(input_np)))


class TestInverseAPIError(unittest.TestCase):
    def test_errors(self):
        input_np = np.random.random([4, 4]).astype("float64")

        self.assertRaises(TypeError, paddle.inverse, input_np)

        for dtype in ["bool", "int32", "int64", "float16"]:
            input = fluid.data(name='input_' + dtype, shape=[4, 4], dtype=dtype)
            self.assertRaises(TypeError, paddle.inverse, input)

        input = fluid.data(name='input_1', shape=[4, 4], dtype="float32")
        out = fluid.data(name='output', shape=[4, 4], dtype="float64")
        self.assertRaises(TypeError, paddle.inverse, input, out)

        input = fluid.data(name='input_2', shape=[4], dtype="float32")
        self.assertRaises(ValueError, paddle.inverse, input)


if __name__ == "__main__":
    unittest.main()
