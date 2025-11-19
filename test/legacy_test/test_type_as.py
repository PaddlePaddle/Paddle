# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base


def api_warpprt(x, y):
    return x.type_as(y)


class TestTypeAsBase(unittest.TestCase):
    def setUp(self):
        self.input_dtype_1 = "float32"
        self.input_dtype_2 = "float16"
        self.input_shape = (2, 3)

        self.input_np_1 = self.generate_data(
            self.input_dtype_1, self.input_shape
        )
        self.input_np_2 = self.generate_data(
            self.input_dtype_2, self.input_shape
        )

        self.input_shape_1 = self.input_np_1.shape
        self.input_shape_2 = self.input_np_2.shape

        self.op_static = api_warpprt
        self.op_dygraph = api_warpprt
        self.places = [None, paddle.CPUPlace()]

    def generate_data(self, dtype, shape):
        if "int" in dtype:
            data = np.arange(1, np.prod(shape) + 1).reshape(shape)
        else:
            data = np.arange(1, np.prod(shape) + 1, dtype='float32').reshape(
                shape
            )
        return data.astype(dtype)

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input_name_1 = 'input_1'
            input_name_2 = 'input_2'
            input_var_1 = paddle.static.data(
                name=input_name_1,
                shape=self.input_shape_1,
                dtype=self.input_dtype_1,
            )
            input_var_2 = paddle.static.data(
                name=input_name_2,
                shape=self.input_shape_2,
                dtype=self.input_dtype_2,
            )
            res = self.op_static(input_var_1, input_var_2)
            exe = base.Executor(place)
            fetches = exe.run(
                main_prog,
                feed={
                    input_name_1: self.input_np_1,
                    input_name_2: self.input_np_2,
                },
                fetch_list=[res],
            )
            self.assertEqual(fetches[0].dtype, np.dtype(self.input_dtype_2))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def check_dygraph_result(self, place):
        with base.dygraph.guard(place):
            input_1 = paddle.to_tensor(self.input_np_1)
            input_2 = paddle.to_tensor(self.input_np_2)
            result = self.op_dygraph(input_1, input_2)
            self.assertEqual(result.dtype, input_2.dtype)

    def test_dygraph(self):
        for place in self.places:
            self.check_dygraph_result(place=place)


class TestTypeAsFloat32ToFloat16(TestTypeAsBase):
    def setUp(self):
        self.input_dtype_1 = "float32"
        self.input_dtype_2 = "float16"
        super().setUp()


class TestTypeAsFloat64ToFloat32(TestTypeAsBase):
    def setUp(self):
        self.input_dtype_1 = "float64"
        self.input_dtype_2 = "float32"
        super().setUp()


class TestTypeAsInt32ToInt64(TestTypeAsBase):
    def setUp(self):
        self.input_dtype_1 = "int32"
        self.input_dtype_2 = "int64"
        super().setUp()


class TestTypeAsInt32ToFloat32(TestTypeAsBase):
    def setUp(self):
        self.input_dtype_1 = "int32"
        self.input_dtype_2 = "float32"
        super().setUp()


class TestTypeAsFloat32ToInt64(TestTypeAsBase):
    def setUp(self):
        self.input_dtype_1 = "float32"
        self.input_dtype_2 = "int64"
        super().setUp()


class TestTypeAsInt8ToFloat64(TestTypeAsBase):
    def setUp(self):
        self.input_dtype_1 = "int8"
        self.input_dtype_2 = "float64"
        self.input_shape = (4, 2)
        super().setUp()


class TestTypeAsUInt8ToInt32(TestTypeAsBase):
    def setUp(self):
        self.input_dtype_1 = "uint8"
        self.input_dtype_2 = "int32"
        self.input_shape = (3, 3)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
