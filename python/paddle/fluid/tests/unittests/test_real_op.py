#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.static as static
from op_test import OpTest


class TestRealOp(OpTest):
    def setUp(self):
        # switch to static
        paddle.enable_static()
        # op test config
        self.op_type = "real"
        self.dtype = np.float64
        self.inputs = {
            'X': np.random.random(
                (20, 5)).astype(self.dtype) + 1j * np.random.random(
                    (20, 5)).astype(self.dtype)
        }
        self.outputs = {'Out': np.real(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()


class TestRealAPI(unittest.TestCase):
    def setUp(self):
        # switch to static
        paddle.enable_static()
        # prepare test attrs
        self._dtypes = ["complex64", "complex128"]
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))
        self._shape = [2, 3]

    def test_in_static_mode(self):
        def init_input_output(dtype):
            input = np.random.random(self._shape).astype(
                dtype) + 1j * np.random.random(self._shape).astype(dtype)
            return {'x': input}, np.real(input)

        for dtype in self._dtypes:
            input_dict, np_res = init_input_output(dtype)
            for place in self._places:
                with static.program_guard(static.Program()):
                    x = static.data(name="x", shape=self._shape, dtype=dtype)
                    out = paddle.real(x)

                    exe = static.Executor(place)
                    out_value = exe.run(feed=input_dict, fetch_list=[out.name])
                    self.assertTrue(np.array_equal(np_res, out_value[0]))

    def test_in_dynamic_mode(self):
        for dtype in self._dtypes:
            input = np.random.random(self._shape).astype(
                dtype) + 1j * np.random.random(self._shape).astype(dtype)
            np_res = np.real(input)
            for place in self._places:
                # it is more convenient to use `guard` than `enable/disable_**` here
                with fluid.dygraph.guard(place):
                    input_t = paddle.to_tensor(input)
                    res = paddle.real(input_t).numpy()
                    self.assertTrue(np.array_equal(np_res, res))
                    res_t = input_t.real().numpy()
                    self.assertTrue(np.array_equal(np_res, res_t))

    def test_name_argument(self):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=self._shape, dtype=self._dtypes[0])
            out = paddle.real(x, name="real_res")
            self.assertTrue("real_res" in out.name)

    def test_dtype_error(self):
        # in static mode
        with self.assertRaises(TypeError):
            with static.program_guard(static.Program()):
                x = static.data(name="x", shape=self._shape, dtype="float32")
                out = paddle.real(x, name="real_res")

        # in dynamic mode
        with self.assertRaises(RuntimeError):
            with fluid.dygraph.guard():
                input = np.random.random(self._shape).astype("float32")
                input_t = paddle.to_tensor(input)
                res = paddle.real(input_t)


if __name__ == "__main__":
    unittest.main()
