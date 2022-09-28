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
import paddle
import paddle.fluid as fluid
from paddle.fluid.data_feeder import convert_dtype
import paddle.fluid.core as core
from paddle.static import program_guard, Program


class TestEmptyLikeAPICommon(unittest.TestCase):

    def __check_out__(self, out):
        data_type = convert_dtype(out.dtype)
        self.assertEqual(
            data_type, self.dst_dtype,
            'dtype should be %s, but get %s' % (self.dst_dtype, data_type))

        shape = out.shape
        self.assertTupleEqual(
            shape, self.dst_shape,
            'shape should be %s, but get %s' % (self.dst_shape, shape))

        if data_type in ['float32', 'float64', 'int32', 'int64']:
            max_value = np.nanmax(out)
            min_value = np.nanmin(out)
            always_non_full_zero = max_value >= min_value
            always_full_zero = max_value == 0.0 and min_value == 0.0
            self.assertTrue(always_full_zero or always_non_full_zero,
                            'always_full_zero or always_non_full_zero.')
        elif data_type in ['bool']:
            total_num = out.size
            true_num = np.sum(out == True)
            false_num = np.sum(out == False)
            self.assertTrue(total_num == true_num + false_num,
                            'The value should always be True or False.')
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

        dtype = 'float32'

        train_program = Program()
        startup_program = Program()

        with program_guard(train_program, startup_program):
            x = np.random.random(self.x_shape).astype(dtype)
            data_x = paddle.static.data('x',
                                        shape=self.data_x_shape,
                                        dtype=dtype)

            out = paddle.empty_like(data_x)

        place = paddle.CUDAPlace(
            0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        res = exe.run(train_program, feed={'x': x}, fetch_list=[out])

        self.dst_dtype = dtype
        self.dst_shape = x.shape
        self.__check_out__(res[0])

        paddle.disable_static()

    def init_config(self):
        self.x_shape = (200, 3)
        self.data_x_shape = [200, 3]


class TestEmptyLikeAPI_Static2(TestEmptyLikeAPI_Static):

    def init_config(self):
        self.x_shape = (3, 200, 3)
        self.data_x_shape = [-1, 200, 3]


class TestEmptyError(unittest.TestCase):

    def test_attr(self):

        def test_dtype():
            x = np.random.random((200, 3)).astype("float64")
            dtype = 'uint8'
            result = paddle.empty_like(x, dtype=dtype)

        self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    unittest.main()
