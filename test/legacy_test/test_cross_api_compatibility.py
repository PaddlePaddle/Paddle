# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


class TestCrossAPI(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype='float32'
        )
        self.y = paddle.to_tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype='float32'
        )
        self.expected = np.cross(self.x.numpy(), self.y.numpy(), axis=1)

    def test_standard(self):
        z = paddle.cross(self.x, self.y, axis=1)
        np.testing.assert_allclose(z.numpy(), self.expected, rtol=1e-5)

    def test_aliases(self):
        z = paddle.cross(input=self.x, other=self.y, dim=1)
        np.testing.assert_allclose(z.numpy(), self.expected, rtol=1e-5)

    def test_default_axis(self):
        z = paddle.cross(self.x, self.y)
        self.assertEqual(z.shape, [3, 3])

    def test_out(self):
        out = paddle.zeros_like(self.x)
        z = paddle.cross(self.x, self.y, axis=1, out=out)
        np.testing.assert_allclose(out.numpy(), self.expected, rtol=1e-5)
        self.assertTrue(z is out)

    def test_mixed_args(self):
        z = paddle.cross(self.x, other=self.y, dim=1)
        np.testing.assert_allclose(z.numpy(), self.expected, rtol=1e-5)

    def test_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[3, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[3, 3], dtype='float32')
            z = paddle.cross(x, y, axis=1)
            exe = paddle.static.Executor(paddle.CPUPlace())
            out = exe.run(
                feed={'x': self.x.numpy(), 'y': self.y.numpy()}, fetch_list=[z]
            )
            np.testing.assert_allclose(out[0], self.expected, rtol=1e-5)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
