#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle


class TestAddnOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        l = 32
        self.x_np = np.random.random([l, 16, 256])

    def check_main(self, x_np, dtype, axis=None):
        paddle.disable_static()
        x = []
        for i in range(x_np.shape[0]):
            val = paddle.to_tensor(x_np[i].astype(dtype))
            val.stop_gradient = False
            x.append(val)
        y = paddle.add_n(x)
        x_g = paddle.grad(y, x)
        y_np = y.numpy().astype('float32')
        x_g_np = []
        for val in x_g:
            x_g_np.append(val.numpy().astype('float32'))
        paddle.enable_static()
        return y_np, x_g_np

    def test_add_n_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        y_np_16, x_g_np_16 = self.check_main(self.x_np, 'float16')
        y_np_32, x_g_np_32 = self.check_main(self.x_np, 'float32')

        np.testing.assert_allclose(y_np_16, y_np_32, rtol=1e-03)
        for i in range(len(x_g_np_32)):
            np.testing.assert_allclose(x_g_np_16[i], x_g_np_32[i], rtol=1e-03)

    def test_add_n_api(self):
        if not paddle.is_compiled_with_cuda():
            return

        y_np_32, x_g_np_32 = self.check_main(self.x_np, 'float32')
        y_np_gt = np.sum(self.x_np, axis=0).astype('float32')

        np.testing.assert_allclose(y_np_32, y_np_gt, rtol=1e-06)


if __name__ == "__main__":
    unittest.main()
