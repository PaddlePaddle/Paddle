#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import unittest
import numpy as np


class TestAllcloseLayer(unittest.TestCase):
    def allclose_check(self, use_cuda):
        a = fluid.data(name="a", shape=[2], dtype='float32')
        b = fluid.data(name="b", shape=[2], dtype='float32')

        result = paddle.allclose(
            a, b, rtol=1e-05, atol=1e-08, equal_nan=False, name="ignore_nan")
        result_nan = paddle.allclose(
            a, b, rtol=1e-05, atol=1e-08, equal_nan=True, name="equal_nan")

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        x = np.array([10000., 1e-07]).astype("float32")
        y = np.array([10000.1, 1e-08]).astype("float32")
        result_v, result_nan_v = exe.run(feed={'a': x,
                                               'b': y},
                                         fetch_list=[result, result_nan])
        self.assertEqual(result_v[0], False)
        self.assertEqual(result_nan_v[0], False)

        x = np.array([10000., 1e-08]).astype("float32")
        y = np.array([10000.1, 1e-09]).astype("float32")
        result_v, result_nan_v = exe.run(feed={'a': x,
                                               'b': y},
                                         fetch_list=[result, result_nan])
        self.assertEqual(result_v[0], True)
        self.assertEqual(result_nan_v[0], True)

        x = np.array([1.0, float('nan')]).astype("float32")
        y = np.array([1.0, float('nan')]).astype("float32")
        result_v, result_nan_v = exe.run(feed={'a': x,
                                               'b': y},
                                         fetch_list=[result, result_nan])
        self.assertEqual(result_v[0], False)
        self.assertEqual(result_nan_v[0], True)

    def test_allclose_cpu(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                self.allclose_check(use_cuda=False)

    def test_allclose_gpu(self):
        if fluid.core.is_compiled_with_cuda():
            main = fluid.Program()
            startup = fluid.Program()
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    self.allclose_check(use_cuda=True)

    def test_dygraph_mode(self):
        x_1 = np.array([10000., 1e-07]).astype("float32")
        y_1 = np.array([10000.1, 1e-08]).astype("float32")
        x_2 = np.array([10000., 1e-08]).astype("float32")
        y_2 = np.array([10000.1, 1e-09]).astype("float32")
        x_3 = np.array([1.0, float('nan')]).astype("float32")
        y_3 = np.array([1.0, float('nan')]).astype("float32")

        with fluid.dygraph.guard():
            x_v_1 = fluid.dygraph.to_variable(x_1)
            y_v_1 = fluid.dygraph.to_variable(y_1)
            ret_1 = paddle.allclose(
                x_v_1,
                y_v_1,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
                name='test_1')
            self.assertEqual(ret_1.numpy()[0], False)
            ret_1 = paddle.allclose(
                x_v_1,
                y_v_1,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=True,
                name='test_2')
            self.assertEqual(ret_1.numpy()[0], False)
            x_v_2 = fluid.dygraph.to_variable(x_2)
            y_v_2 = fluid.dygraph.to_variable(y_2)
            ret_2 = paddle.allclose(
                x_v_2,
                y_v_2,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
                name='test_3')
            self.assertEqual(ret_2.numpy()[0], True)
            ret_2 = paddle.allclose(
                x_v_2,
                y_v_2,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=True,
                name='test_4')
            self.assertEqual(ret_2.numpy()[0], True)
            x_v_3 = fluid.dygraph.to_variable(x_3)
            y_v_3 = fluid.dygraph.to_variable(y_3)
            ret_3 = paddle.allclose(
                x_v_3,
                y_v_3,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
                name='test_5')
            self.assertEqual(ret_3.numpy()[0], False)
            ret_3 = paddle.allclose(
                x_v_3,
                y_v_3,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=True,
                name='test_6')
            self.assertEqual(ret_3.numpy()[0], True)


if __name__ == "__main__":
    unittest.main()
