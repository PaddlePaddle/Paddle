#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestReduceOp_Stride(unittest.TestCase):
    def setUp(self):
        self.python_api = paddle.max
        self.numpy_api = np.max

    def init_dtype(self):
        self.dtype = np.float64

    def init_place(self):
        self.place = core.CUDAPlace(0)

    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = self.numpy_api(self.x)
        self.perm = [1, 0]
        self.x_trans = np.transpose(self.x, self.perm)

    def test_dynamic_api(self):
        self.init_dtype()
        self.init_place()
        self.init_input_output()
        paddle.disable_static()
        self.pd_x_trans = paddle.to_tensor(self.x_trans, place=self.place)
        if self.strided_input_type == "transpose":
            x_trans_tmp = paddle.transpose(self.pd_x_trans, self.perm)
        elif self.strided_input_type == "as_stride":
            x_trans_tmp = paddle.as_strided(
                self.pd_x_trans, self.shape_param, self.stride_param
            )
        else:
            raise TypeError(f"Unsupported test type {self.strided_input_type}.")
        res = self.python_api(x_trans_tmp)
        res = res.cpu().numpy()
        np.testing.assert_allclose(res, self.out, rtol=1e-05)


def create_test_act_stride_class(base_class, api_name, paddle_api, numpy_api):
    class TestStride1(base_class):
        def setUp(self):
            self.python_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.out = self.numpy_api(self.x)
            self.perm = [0, 1, 3, 2]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride1")
    TestStride1.__name__ = cls_name
    globals()[cls_name] = TestStride1

    class TestStride2(base_class):
        def setUp(self):
            self.python_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.out = self.numpy_api(self.x)
            self.perm = [0, 2, 1, 3]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride2")
    TestStride2.__name__ = cls_name
    globals()[cls_name] = TestStride2

    class TestStride3(base_class):
        def setUp(self):
            self.python_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.out = self.numpy_api(self.x)
            self.perm = [0, 1, 3, 2]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride3")
    TestStride3.__name__ = cls_name
    globals()[cls_name] = TestStride3

    class TestStride4(base_class):
        def setUp(self):
            self.python_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [1, 2, 13, 17]).astype(
                self.dtype
            )
            self.out = self.numpy_api(self.x)
            self.perm = [1, 0, 2, 3]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride4")
    TestStride4.__name__ = cls_name
    globals()[cls_name] = TestStride4

    class TestStride5(base_class):
        def setUp(self):
            self.python_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "as_stride"
            self.x = np.random.uniform(0.1, 1, [23, 2, 13, 20]).astype(
                self.dtype
            )
            self.x_trans = self.x
            self.x = self.x[:, 0:1, :, 0:1]
            self.out = self.numpy_api(self.x)
            self.shape_param = [23, 1, 13, 1]
            self.stride_param = [520, 260, 20, 1]

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride5")
    TestStride5.__name__ = cls_name
    globals()[cls_name] = TestStride5

    class TestStrideZeroSize1(base_class):
        def setUp(self):
            self.python_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.rand(1, 0, 2).astype('float32')
            self.out = self.numpy_api(self.x)
            self.perm = [2, 1, 0]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(
        base_class.__name__, api_name, "StrideZeroSize1"
    )
    TestStrideZeroSize1.__name__ = cls_name
    globals()[cls_name] = TestStrideZeroSize1


create_test_act_stride_class(TestReduceOp_Stride, "Max", paddle.max, np.max)

create_test_act_stride_class(TestReduceOp_Stride, "Min", paddle.min, np.min)

create_test_act_stride_class(TestReduceOp_Stride, "Amax", paddle.amax, np.amax)

create_test_act_stride_class(TestReduceOp_Stride, "Amin", paddle.amin, np.amin)

create_test_act_stride_class(TestReduceOp_Stride, "Sum", paddle.sum, np.sum)

create_test_act_stride_class(TestReduceOp_Stride, "Mean", paddle.mean, np.mean)

create_test_act_stride_class(TestReduceOp_Stride, "Prod", paddle.prod, np.prod)

create_test_act_stride_class(TestReduceOp_Stride, "All", paddle.all, np.all)

create_test_act_stride_class(TestReduceOp_Stride, "Any", paddle.any, np.any)

if __name__ == '__main__':
    unittest.main()
