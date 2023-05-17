# # Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

import unittest

import numpy as np

import paddle


def ref_cdist(x, y, p=2.0):
    r1 = x.shape[-2]
    r2 = y.shape[-2]
    if r1 == 0 or r2 == 0:
        return np.empty((r1, r2), x.dtype)
    return np.linalg.norm(x[..., None, :] - y[..., None, :, :], ord=p, axis=-1)


class TestCdistAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(1024)
        self.x_shape = [11, 12]
        self.y_shape = [10, 12]
        self.x = np.random.uniform(-3, 3, self.x_shape).astype('float32')
        self.y = np.random.uniform(-3, 3, self.y_shape).astype('float32')
        self.p = 2.0
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x_shape)
            y = paddle.static.data('y', self.y_shape)
            out = paddle.cdist(x, y, p=self.p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.x, 'y': self.y},
                fetch_list=[
                    out,
                ],
            )
        out_ref = ref_cdist(self.x, self.y, p=self.p)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.cdist(x, y, p=self.p)
        out_ref = ref_cdist(self.x, self.y, p=self.p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)
        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
