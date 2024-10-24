#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestFillAnyOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fill_any'
        self.use_dynamic_create_class = False

    class TestFillAnyOp(XPUOpTest):
        def setUp(self):
            self.op_type = "fill_any"
            self.dtype = self.in_type
            self.value = 0.0
            self.init()
            self.inputs = {'X': np.random.random((20, 30)).astype(self.dtype)}
            self.attrs = {'value': float(self.value)}
            out_np = self.value * np.ones_like(self.inputs["X"])
            if self.dtype == np.uint16:
                out_np = convert_float_to_uint16(out_np)
            else:
                out_np = out_np.astype(self.dtype)
            self.outputs = {'Out': out_np}

        def init(self):
            pass

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')

    class TestFillAnyOpvalue1(TestFillAnyOp):
        def init(self):
            self.value = 11555

    class TestFillAnyOpvalue2(TestFillAnyOp):
        def init(self):
            self.value = 11111.1111


class TestFillAnyInplace(unittest.TestCase):
    def test_fill_any_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
            self.assertEqual(var.inplace_version, 0)

            var.fill_(0)
            self.assertEqual(var.inplace_version, 1)

            var.fill_(0)
            self.assertEqual(var.inplace_version, 2)

            var.fill_(0)
            self.assertEqual(var.inplace_version, 3)

    def test_fill_any_equal(self):
        with paddle.base.dygraph.guard():
            tensor = paddle.to_tensor(
                np.random.random((20, 30)).astype(np.float32)
            )
            target = tensor.numpy()
            target[...] = 1

            tensor.fill_(1)
            self.assertEqual((tensor.numpy() == target).all().item(), True)

    def test_backward(self):
        with paddle.base.dygraph.guard():
            x = paddle.full([10, 10], -1.0, dtype='float32')
            x.stop_gradient = False
            y = 2 * x
            y.fill_(1)
            y.backward()
            np.testing.assert_array_equal(x.grad.numpy(), np.zeros([10, 10]))


class TestFillAnyLikeOpSpecialValue(unittest.TestCase):
    def setUp(self):
        self.special_values = [float("nan"), float("+inf"), float("-inf")]
        self.dtypes = ["float32", "float16"]

    def test_dygraph_api(self):
        paddle.disable_static()
        paddle.set_device("xpu")
        for dtype in self.dtypes:
            for value in self.special_values:
                ref = paddle.empty([4, 4], dtype=dtype)
                val_pd = paddle.full_like(ref, value, dtype=dtype)
                val_np = np.full([4, 4], value, dtype=dtype)
                np.testing.assert_equal(val_pd.numpy(), val_np)
        paddle.enable_static()


support_types = get_xpu_op_support_types('fill_any')
for stype in support_types:
    create_test_class(globals(), XPUTestFillAnyOp, stype)

if __name__ == "__main__":
    unittest.main()
