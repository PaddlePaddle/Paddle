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

import numpy as np
import unittest
import sys

sys.path.append("..")
from eager_op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestMul(OpTest):
    # case 1: (32, 5) * (5, 100) -> (32, 100)
    def config(self):
        self.x_shape = (32, 5)
        self.y_shape = (5, 100)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype),
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ['X', 'Y'],
            'Out',
            max_relative_error=0.0065,
        )

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            max_relative_error=0.0065,
        )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            no_grad_set=set("Y"),
            max_relative_error=0.0065,
        )


@skip_check_grad_ci(
    reason="Don't support grad checking for NPU OP with FP16 data type."
)
class TestMulFP16(TestMul):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestMul2(TestMul):
    # case 2: (20, 2, 5) * (10, 50) -> (20, 50), x_num_col_dims = 1
    def config(self):
        self.x_shape = (20, 2, 5)
        self.y_shape = (10, 50)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype),
        }
        self.outputs = {
            'Out': np.dot(self.inputs['X'].reshape(20, 10), self.inputs['Y'])
        }


@skip_check_grad_ci(
    reason="Don't support grad checking for NPU OP with FP16 data type."
)
class TestMul2FP16(TestMul2):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestMul3(TestMul):
    # case 3: (20, 3, 4) * (4, 50) -> (20, 3, 50), x_num_col_dims = 2

    def config(self):
        self.x_shape = (20, 3, 4)
        self.y_shape = (4, 50)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype),
        }
        self.attrs = {"x_num_col_dims": 2}
        self.outputs = {'Out': np.matmul(self.inputs['X'], self.inputs['Y'])}


@skip_check_grad_ci(
    reason="Don't support grad checking for NPU OP with FP16 data type."
)
class TestMul3FP16(TestMul3):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestMul4(TestMul):
    # case 4: (20, 2, 2, 3) * (12, 50) -> (20, 50), x_num_col_dims = 1
    def config(self):
        self.x_shape = (20, 2, 2, 3)
        self.y_shape = (12, 50)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype),
        }
        self.outputs = {
            'Out': np.dot(self.inputs['X'].reshape(20, 12), self.inputs['Y'])
        }


@skip_check_grad_ci(
    reason="Don't support grad checking for NPU OP with FP16 data type."
)
class TestMul4FP16(TestMul4):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


if __name__ == '__main__':
    unittest.main()
