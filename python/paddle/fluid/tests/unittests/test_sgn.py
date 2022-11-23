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

import unittest
import numpy as np
import paddle


def np_sgn(x: np.ndarray):
    if x.dtype == 'complex128' or x.dtype == 'complex64':
        x_abs = np.abs(x)
        eps = np.finfo(x.dtype).eps
        x_abs = np.maximum(x_abs, eps)
        out = x / x_abs
    else:
        out = np.sign(x)
    return out


class TestSgnError(unittest.TestCase):

    def test_errors(self):
        # The input dtype of sgn must be float16, float32, float64,complex64,complex128.
        input2 = paddle.to_tensor(
            np.random.randint(-10, 10, size=[12, 20]).astype('int32'))
        input3 = paddle.to_tensor(
            np.random.randint(-10, 10, size=[12, 20]).astype('int64'))

        self.assertRaises(TypeError, paddle.sgn, input2)
        self.assertRaises(TypeError, paddle.sgn, input3)


class TestSignAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.support_dtypes = [
            'float16', 'float32', 'float64', 'complex64', 'complex128'
        ]
        if paddle.device.get_device() == 'cpu':
            self.support_dtypes = [
                'float32', 'float64', 'complex64', 'complex128'
            ]

    def test_dtype(self):
        for dtype in self.support_dtypes:
            x = paddle.to_tensor(
                np.random.randint(-10, 10, size=[12, 20, 2]).astype(dtype))

            paddle.sgn(x)

    def test_complex(self):
        for dtype in ['complex64', 'complex128']:
            np_x = np.array([[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]],
                            dtype=dtype)
            x = paddle.to_tensor(np_x)
            z = paddle.sgn(x)
            np_z = z.numpy()
            z_expected = np_sgn(np_x)
            np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)

    def test_float(self):
        for dtype in self.support_dtypes:
            np_x = np.random.randint(-10, 10, size=[12, 20, 2]).astype(dtype)
            x = paddle.to_tensor(np_x)
            z = paddle.sgn(x)
            np_z = z.numpy()
            z_expected = np_sgn(np_x)
            np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
