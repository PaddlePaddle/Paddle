# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from operator import __add__, __sub__, __mul__, __truediv__

import numpy as np
import paddle
# import paddle.sparse as sparse

op_list = [__truediv__]


def get_actual_res(x, y, op):
    if op == __truediv__:
        res = paddle.sparse.divide(x, y)
    else:
        raise ValueError("unsupported op")
    return res


class TestSparseElementWiseAPI(unittest.TestCase):
    """
    test paddle.sparse.add, subtract, multiply, divide
    """

    def setUp(self):
        paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        paddle.fluid.set_flags({"FLAGS_use_system_allocator": True})
        np.random.seed(2022)
        self.op_list = op_list
        self.csr_shape = [128, 256]  # [0,0]～ [100,100] 都不会挂
        self.coo_shape = [4, 8, 3, 5]

        # 下面是各种搭配情况
        # self.support_dtypes = ['int32']
        # self.support_dtypes = ['int64']
        # self.support_dtypes = ['float32']
        # self.support_dtypes = ['float64']
        self.support_dtypes = ['int32', 'int64']  # double free or corruption (!prev) / corrupted double-linked list
        # self.support_dtypes = ['int32', 'float32']
        # self.support_dtypes = ['int32', 'float64']
        # self.support_dtypes = ['int32', 'int32'] # free(): corrupted unsorted chunks

        # self.support_dtypes = ['int64', 'int32'] #  free(): corrupted unsorted chunks
        # self.support_dtypes = ['int64', 'float32']
        # self.support_dtypes = ['int64', 'float64']
        # self.support_dtypes = ['int64', 'int64'] # corrupted double-linked list

        # self.support_dtypes = ['float32', 'float64']
        # self.support_dtypes = ['float32', 'int32'] # free(): corrupted unsorted chunks
        # self.support_dtypes = ['float32', 'int64'] # corrupted size vs. prev_size
        # self.support_dtypes = ['float32', 'float32']

        # self.support_dtypes = ['float64', 'int32'] # free(): corrupted unsorted chunks
        # self.support_dtypes = ['float64', 'float32']
        # self.support_dtypes = ['float64', 'int64'] # double free or corruption (!prev)
        # self.support_dtypes = ['float64', 'float64']

        # 原始代码
        # self.support_dtypes = ['float32', 'float64', 'int32', 'int64']

    def func_test_csr(self, op):
        for dtype in self.support_dtypes:
            x = np.random.randint(-255, 255, size=self.csr_shape).astype(dtype)
            y = np.random.randint(-255, 255, size=self.csr_shape).astype(dtype)

            dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
            dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)

            s_dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
            s_dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)
            csr_x = s_dense_x.to_sparse_csr()
            csr_y = s_dense_y.to_sparse_csr()

            actual_res = get_actual_res(csr_x, csr_y, op)
            actual_res.backward(actual_res)

    def test_support_dtypes_csr(self):
        paddle.device.set_device('cpu')
        if paddle.device.get_device() == "cpu":
            for op in op_list:
                self.func_test_csr(op)


if __name__ == "__main__":
    paddle.device.set_device('cpu')
    unittest.main()
