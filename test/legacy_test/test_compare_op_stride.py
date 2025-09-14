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


@unittest.skipIf(
    not paddle.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestBinaryElementwiseOp_Stride(unittest.TestCase):
    def setUp(self):
        self.place = paddle.core.CUDAPlace(0)
        self.dtype = np.float64
        self.init_api()
        self.init_input()

    def init_api(self):
        self.paddle_api = paddle.less_than
        self.numpy_api = np.less

    def init_input(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.perm = [1, 0]
        self.x_trans = np.transpose(self.x, self.perm)

    def test_dygraph_api_arithmetic(self):
        paddle.disable_static()
        x_trans = paddle.to_tensor(self.x_trans, place=self.place)
        y = paddle.to_tensor(self.y, place=self.place)
        if self.strided_input_type == "transpose":
            x_non_conti = paddle.transpose(x_trans, self.perm)
        elif self.strided_input_type == "as_stride":
            x_non_conti = paddle.as_strided(
                x_trans, self.shape_param, self.stride_param
            )
        else:
            raise TypeError(f"Unsupported test type {self.strided_input_type}.")
        out = self.paddle_api(x_non_conti, y)
        out_ref = self.numpy_api(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy())
        paddle.enable_static()


def create_test_act_stride_class(base_class, api_name, paddle_api, numpy_api):
    class TestStride1(base_class):
        def init_api(self):
            self.paddle_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.perm = [0, 1, 3, 2]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride1")
    TestStride1.__name__ = cls_name
    globals()[cls_name] = TestStride1

    class TestStride2(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.perm = [0, 2, 1, 3]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride2")
    TestStride2.__name__ = cls_name
    globals()[cls_name] = TestStride2

    class TestStride3(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(
                self.dtype
            )
            self.perm = [0, 1, 3, 2]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride3")
    TestStride3.__name__ = cls_name
    globals()[cls_name] = TestStride3

    class TestStride4(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [1, 2, 13, 17]).astype(
                self.dtype
            )
            self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(
                self.dtype
            )
            self.perm = [1, 0, 2, 3]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride4")
    TestStride4.__name__ = cls_name
    globals()[cls_name] = TestStride4

    class TestStride5(base_class):
        def init_input(self):
            self.strided_input_type = "as_stride"
            self.x = np.random.uniform(0.1, 1, [23, 2, 13, 20]).astype(
                self.dtype
            )
            self.y = np.random.uniform(0.1, 1, [23, 10, 1, 17]).astype(
                self.dtype
            )
            self.x_trans = self.x
            self.x = self.x[:, 0:1, :, 0:1]
            self.shape_param = [23, 1, 13, 1]
            self.stride_param = [520, 260, 20, 1]

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride5")
    TestStride5.__name__ = cls_name
    globals()[cls_name] = TestStride5

    class TestStrideZeroDim1(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
            self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.perm = []
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(
        base_class.__name__, api_name, "StrideZeroDim1"
    )
    TestStrideZeroDim1.__name__ = cls_name
    globals()[cls_name] = TestStrideZeroDim1

    class TestStrideZeroSize1(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.rand(1, 0, 2).astype('float32')
            self.y = np.random.rand(3, 0, 1).astype('float32')
            self.perm = [2, 1, 0]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(
        base_class.__name__, api_name, "StrideZeroSize1"
    )
    TestStrideZeroSize1.__name__ = cls_name
    globals()[cls_name] = TestStrideZeroSize1


create_test_act_stride_class(
    TestBinaryElementwiseOp_Stride, "Lessthan", paddle.less_than, np.less
)
create_test_act_stride_class(
    TestBinaryElementwiseOp_Stride,
    "Lessequal",
    paddle.less_equal,
    np.less_equal,
)
create_test_act_stride_class(
    TestBinaryElementwiseOp_Stride,
    "Greaterthan",
    paddle.greater_than,
    np.greater,
)
create_test_act_stride_class(
    TestBinaryElementwiseOp_Stride,
    "Greaterequal",
    paddle.greater_equal,
    np.greater_equal,
)
create_test_act_stride_class(
    TestBinaryElementwiseOp_Stride, "Equal", paddle.equal, np.equal
)
create_test_act_stride_class(
    TestBinaryElementwiseOp_Stride, "Notequal", paddle.not_equal, np.not_equal
)

if __name__ == "__main__":
    unittest.main()
