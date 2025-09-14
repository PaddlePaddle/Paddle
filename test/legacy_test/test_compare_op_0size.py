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

import numpy

import paddle


def create_test_class(op_type, dtype):
    class Cls(unittest.TestCase):
        def setUp(self):
            pass

        def test_0size(self):
            numpy_tensor_x = numpy.ones([1]).astype(dtype)
            paddle_x = paddle.to_tensor(numpy_tensor_x)
            paddle_x.stop_gradient = False
            numpy_tensor_x2 = numpy.ones([0]).astype(dtype)
            paddle_x2 = paddle.to_tensor(numpy_tensor_x2)
            paddle_x2.stop_gradient = False

            paddle_api = eval(f"paddle.{op_type}")
            paddle_out = paddle_api(paddle_x, paddle_x2)
            numpy_api = eval(f"numpy.{op_type}")

            numpy.testing.assert_allclose(
                paddle_out.numpy(),
                numpy_api(numpy_tensor_x, numpy_tensor_x2),
                1e-2,
                1e-2,
            )

    cls_name = f"{op_type}_0SizeTest"
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


create_test_class("equal", "float32")
create_test_class("bitwise_or", "int32")
create_test_class("bitwise_xor", "int32")
create_test_class("bitwise_and", "int32")
create_test_class("logical_or", "float32")
create_test_class("logical_xor", "float32")
create_test_class("logical_and", "float32")
if not paddle.base.core.is_compiled_with_xpu():
    create_test_class("bitwise_left_shift", "int32")
    create_test_class("bitwise_right_shift", "int32")

if __name__ == '__main__':
    unittest.main()
