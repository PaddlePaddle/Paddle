#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_uint16_to_float

import paddle
from paddle.base import core
from paddle.base.data_feeder import convert_dtype


class TestEmptyLikeAPICommon(unittest.TestCase):
    def __check_out__(self, out):
        data_type = convert_dtype(out.dtype)
        self.assertEqual(
            data_type,
            self.dst_dtype,
            f'dtype should be {self.dst_dtype}, but get {data_type}',
        )

        shape = out.shape
        self.assertTupleEqual(
            shape,
            self.dst_shape,
            f'shape should be {self.dst_shape}, but get {shape}',
        )

        if data_type in [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'uint16',
        ]:
            max_value = np.nanmax(out)
            min_value = np.nanmin(out)
            always_non_full_zero = max_value >= min_value
            always_full_zero = max_value == 0.0 and min_value == 0.0
            self.assertTrue(
                always_full_zero or always_non_full_zero,
                'always_full_zero or always_non_full_zero.',
            )
        elif data_type in ['uint16']:
            uout = convert_uint16_to_float(out)
            max_value = np.nanmax(uout)
            min_value = np.nanmin(uout)
            always_non_full_zero = max_value >= min_value
            always_full_zero = max_value == 0.0 and min_value == 0.0
            self.assertTrue(
                always_full_zero or always_non_full_zero,
                'always_full_zero or always_non_full_zero.',
            )
        elif data_type in ['bool']:
            total_num = out.size
            true_num = np.sum(out)
            false_num = np.sum(~out)
            self.assertTrue(
                total_num == true_num + false_num,
                'The value should always be True or False.',
            )
        else:
            self.assertTrue(False, 'invalid data type')


class TestEmptyLikeAPI(TestEmptyLikeAPICommon):
    def setUp(self):
        self.init_config()

    def test_dygraph_api_out(self):
        paddle.disable_static()
        out = paddle.empty_like(self.x, self.dtype)
        self.__check_out__(out.numpy())
        paddle.enable_static()

    def init_config(self):
        self.x = np.random.random((200, 3)).astype("float32")
        self.dtype = self.x.dtype
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI2(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("float64")
        self.dtype = self.x.dtype
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI3(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("int")
        self.dtype = self.x.dtype
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI4(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("int64")
        self.dtype = self.x.dtype
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI5(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("bool")
        self.dtype = self.x.dtype
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI6(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("float64")
        self.dtype = "float32"
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI7(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("int")
        self.dtype = "float32"
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI8(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("int64")
        self.dtype = "float32"
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI9(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("bool")
        self.dtype = "float32"
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI10(TestEmptyLikeAPI):
    def init_config(self):
        self.x = np.random.random((200, 3)).astype("float32")
        self.dtype = "bool"
        self.dst_shape = self.x.shape
        self.dst_dtype = self.dtype


class TestEmptyLikeAPI_Static(TestEmptyLikeAPICommon):
    def setUp(self):
        self.init_config()

    def test_static_graph(self):
        paddle.enable_static()
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(train_program, startup_program):
            x = np.random.random(self.x_shape).astype(self.dtype)
            data_x = paddle.static.data(
                'x', shape=self.data_x_shape, dtype=self.dtype
            )

            out = paddle.empty_like(data_x)

            place = (
                paddle.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            res = exe.run(train_program, feed={'x': x}, fetch_list=[out])

            self.dst_dtype = self.dtype
            self.dst_shape = x.shape
            self.__check_out__(res[0])

            paddle.disable_static()

    def init_config(self):
        self.x_shape = (200, 3)
        self.data_x_shape = [200, 3]
        self.dtype = 'float32'


class TestEmptyLikeAPI_Static2(TestEmptyLikeAPI_Static):
    def init_config(self):
        self.x_shape = (3, 200, 3)
        self.data_x_shape = [-1, 200, 3]
        self.dtype = 'float32'


class TestEmptyLikeAPI_StaticForFP16Op(TestEmptyLikeAPICommon):
    def setUp(self):
        self.init_config()

    def init_config(self):
        self.x_shape = (200, 3)
        self.data_x_shape = [200, 3]
        self.dtype = 'float16'

    def test_static_graph(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = np.random.random([200, 3]).astype(self.dtype)
                data_x = paddle.static.data(
                    name="x", shape=[200, 3], dtype=self.dtype
                )
                out = paddle.empty_like(data_x)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': x},
                    fetch_list=[out],
                )

            self.dst_dtype = self.dtype
            self.dst_shape = x.shape
            self.__check_out__(res[0])


class TestEmptyLikeAPI_StaticForBF16Op(TestEmptyLikeAPICommon):
    def setUp(self):
        self.init_config()

    def init_config(self):
        self.x_shape = (200, 3)
        self.data_x_shape = [200, 3]
        self.dtype = 'uint16'

    def test_static_graph(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = np.random.random([200, 3]).astype(np.uint16)
                data_x = paddle.static.data(
                    name="x", shape=[200, 3], dtype=np.uint16
                )
                out = paddle.empty_like(data_x)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={'x': x},
                    fetch_list=[out],
                )

            self.dst_dtype = self.dtype
            self.dst_shape = x.shape
            self.__check_out__(res[0])


if __name__ == '__main__':
    unittest.main()
