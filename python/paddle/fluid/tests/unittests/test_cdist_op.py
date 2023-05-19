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
        self.x = np.random.rand(10, 20).astype('float32')
        self.y = np.random.rand(11, 20).astype('float32')
        self.p = 2.0
        self.compute_mode = "use_mm_for_euclid_dist_if_necessary"
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        pass

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.cdist(x, y, self.p, self.compute_mode)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            out_ref = ref_cdist(self.x, self.y, self.p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.cdist(x, y, self.p, self.compute_mode)
        out_ref = ref_cdist(self.x, self.y, self.p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)
        paddle.enable_static()


class TestCdistAPICase1(TestCdistAPI):
    def init_input(self):
        self.p = 0


class TestCdistAPICase2(TestCdistAPI):
    def init_input(self):
        self.p = 1.0


class TestCdistAPICase3(TestCdistAPI):
    def init_input(self):
        self.p = 3.0


class TestCdistAPICase4(TestCdistAPI):
    def init_input(self):
        self.p = 1.5


class TestCdistAPICase5(TestCdistAPI):
    def init_input(self):
        self.p = 2.5


class TestCdistAPICase6(TestCdistAPI):
    def init_input(self):
        self.p = float('inf')


class TestCdistAPICase7(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(50, 20).astype('float64')
        self.y = np.random.rand(40, 20).astype('float64')
        self.compute_mode = "use_mm_for_euclid_dist"


class TestCdistAPICase8(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(50, 20).astype('float64')
        self.y = np.random.rand(40, 20).astype('float64')
        self.compute_mode = "donot_use_mm_for_euclid_dist"


class TestCdistAPICase9(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(500, 100).astype('float64')
        self.y = np.random.rand(400, 100).astype('float64')


class TestCdistAPICase10(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(3, 500, 100).astype('float64')
        self.y = np.random.rand(3, 400, 100).astype('float64')


class TestCdistAPICase11(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(3, 4, 500, 100).astype('float64')
        self.y = np.random.rand(3, 4, 400, 100).astype('float64')


class TestCdistAPICase12(TestCdistAPI):
    def init_input(self):
        self.x = np.random.rand(3, 4, 500, 100).astype('float64')
        self.y = np.random.rand(3, 4, 400, 100).astype('float64')
        self.p = 3.0


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
