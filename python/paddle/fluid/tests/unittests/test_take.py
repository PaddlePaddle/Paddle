#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class TestTakeAPI(unittest.TestCase):

    def set_dtype(self):
        self.input_dtype = 'float64'
        self.index_dtype = 'int64'

    def setUp(self):
        self.set_dtype()
        self.place = fluid.CUDAPlace(
            0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
        self.input_shape = [3, 4]
        self.index_shape = [2, 3]
        self.input_np = np.arange(0, 12).reshape(self.input_shape).astype(
            self.input_dtype)
        self.index_np = np.arange(-4, 2).reshape(self.index_shape).astype(
            self.index_dtype)

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = Program()
        train_program = Program()
        with program_guard(startup_program, train_program):
            x = fluid.data(name='input',
                           dtype=self.input_dtype,
                           shape=self.input_shape)
            index = fluid.data(name='index',
                               dtype=self.index_dtype,
                               shape=self.index_shape)
            out = paddle.take(x, index)

            exe = fluid.Executor(self.place)
            st_result = exe.run(fluid.default_main_program(),
                                feed={
                                    'input': self.input_np,
                                    'index': self.index_np
                                },
                                fetch_list=[out])
            self.assertTrue(
                np.allclose(st_result, np.take(self.input_np, self.index_np)))

    def test_dygraph(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.input_np)
        index = paddle.to_tensor(self.index_np)
        dy_result = paddle.take(x, index)
        self.assertTrue(
            np.allclose(np.take(self.input_np, self.index_np),
                        dy_result.numpy()))


class TestTakeInt32(TestTakeAPI):
    """Test take API with data type int32"""

    def set_dtype(self):
        self.input_dtype = 'int32'
        self.index_dtype = 'int64'


class TestTakeInt64(TestTakeAPI):
    """Test take API with data type int64"""

    def set_dtype(self):
        self.input_dtype = 'int64'
        self.index_dtype = 'int64'


class TestTakeFloat32(TestTakeAPI):
    """Test take API with data type float32"""

    def set_dtype(self):
        self.input_dtype = 'float32'
        self.index_dtype = 'int64'


class TestTakeType(TestTakeAPI):
    """Test take Error"""

    def test_static_type_error(self):
        """Argument 'index' must be Tensor"""
        paddle.enable_static()
        with program_guard(Program()):
            x = fluid.data(name='input',
                           dtype=self.input_dtype,
                           shape=self.input_shape)
            self.assertRaises(TypeError, paddle.take, x, self.index_np)

    def test_dygraph_type_error(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.input_np)
        self.assertRaises(TypeError, paddle.take, x, self.index_np)

    def test_static_dtype_error(self):
        """Data type of argument 'index' must be in [paddle.int32, paddle.int64]"""
        paddle.enable_static()
        with program_guard(Program()):
            x = fluid.data(name='input',
                           dtype='float64',
                           shape=self.input_shape)
            index = fluid.data(name='index',
                               dtype='float32',
                               shape=self.index_shape)
            self.assertRaises(TypeError, paddle.take, x, index)

    def test_dygraph_dtype_error(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.input_np)
        index = paddle.to_tensor(self.index_np, dtype='float32')
        self.assertRaises(TypeError, paddle.take, x, index)


if __name__ == "__main__":
    unittest.main()
