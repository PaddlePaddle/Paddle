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


def create_test_class(op_type, dtype, shape):
    class Cls(unittest.TestCase):
        def test_zero_size(self):
            numpy_tensor_1 = np.random.rand(*shape).astype(dtype)
            paddle_x = paddle.to_tensor(numpy_tensor_1)
            paddle_x.stop_gradient = False

            paddle_api = eval(f"paddle.{op_type}")
            paddle_out = paddle_api(paddle_x)
            numpy_api = eval(f"scipy.special.{op_type}")
            numpy_out = numpy_api(numpy_tensor_1)

            np.testing.assert_allclose(
                paddle_out.numpy(),
                numpy_out,
                1e-2,
                1e-2,
            )
            np.testing.assert_allclose(
                paddle_out.shape,
                numpy_out.shape,
            )

    cls_name = f"{op_type}{dtype}_0SizeTest"
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


op_list = ["i0", "i0e", "i1", "i1e"]

for op in op_list:
    create_test_class(op, "float32", [3, 4, 0])
    create_test_class(op, "float64", [3, 4, 0, 3, 4])
    create_test_class(op, "int32", [3, 4, 0])
    create_test_class(op, "int64", [3, 4, 0, 3, 4])

if __name__ == '__main__':
    unittest.main()
