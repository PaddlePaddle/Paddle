#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


def ref_nextafter(x, y):
    out = np.nextafter(x, y)
    return out


class TestNextafterAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.rand(1, 2).astype('float32')
        self.y_np = np.random.rand(1, 2).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                name="X", shape=self.x_np.shape, dtype='float32'
            )
            y = paddle.static.data(
                name="Y", shape=self.y_np.shape, dtype='float32'
            )
            out = paddle.nextafter(x, y)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'X': self.x_np, 'Y': self.y_np}, fetch_list=[out]
            )
        out_ref = ref_nextafter(self.x_np, self.y_np)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.nextafter(x, y)
        out_ref = ref_nextafter(self.x_np, self.y_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
