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

import unittest

import numpy as np
from eager_op_test import OpTest, paddle_static_guard

import paddle


class TestFlattenOp(OpTest):
    def setUp(self):
        self.op_type = "flatten"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.in_shape).astype("float64")}
        self.init_attrs()
        self.outputs = {"Out": self.inputs["X"].reshape(self.new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.in_shape = (3, 2, 2, 10)
        self.axis = 1
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axis": self.axis}


class TestFlattenOp1(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 2, 10)
        self.axis = 0
        self.new_shape = (1, 120)


class TestFlattenOpWithDefaultAxis(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (10, 2, 2, 3)
        self.new_shape = (10, 12)

    def init_attrs(self):
        self.attrs = {}


class TestFlattenOpSixDims(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.axis = 4
        self.new_shape = (36, 16)


class TestFlattenOpFP16(unittest.TestCase):
    def test_fp16_with_gpu(self):
        if paddle.base.core.is_compiled_with_cuda():
            with paddle_static_guard():
                place = paddle.CUDAPlace(0)
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    input = np.random.random([12, 14]).astype("float16")
                    x = paddle.static.data(
                        name="x", shape=[12, 14], dtype="float16"
                    )

                    y = paddle.flatten(x)

                    exe = paddle.static.Executor(place)
                    res = exe.run(
                        paddle.static.default_main_program(),
                        feed={
                            "x": input,
                        },
                        fetch_list=[y],
                    )

                    np.testing.assert_array_equal(res[0].shape, [12 * 14])


if __name__ == "__main__":
    unittest.main()
