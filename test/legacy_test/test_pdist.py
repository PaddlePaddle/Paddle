#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

def ref_pdist(x, p=2.0):
    dist = np.linalg.norm(x[..., None, :] - x[None, :, :], ord=p, axis=-1)
    res = []
    rows, cols = dist.shape
    for i in range(rows):
        for j in range(cols):
            if i >= j: continue
            res.append(dist[i][j])
    return np.array(res)


class TestpdistAPI(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(10, 20).astype('float32')
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
            out = paddle.pdist(x, self.p, self.compute_mode)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x}, fetch_list=[out])
            out_ref = ref_pdist(self.x, self.p)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-5, atol=1e-5)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out = paddle.pdist(x, self.p, self.compute_mode)
        out_ref = ref_pdist(self.x, self.p)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5, atol=1e-5)
        paddle.enable_static()


class TestpdistAPICase1(TestpdistAPI):
    def init_input(self):
        self.p = 0


class TestpdistAPICase2(TestpdistAPI):
    def init_input(self):
        self.p = 1.0


class TestpdistAPICase3(TestpdistAPI):
    def init_input(self):
        self.p = 3.0


class TestpdistAPICase4(TestpdistAPI):
    def init_input(self):
        self.p = 1.5


class TestpdistAPICase5(TestpdistAPI):
    def init_input(self):
        self.p = 2.5


class TestpdistAPICase6(TestpdistAPI):
    def init_input(self):
        self.p = float('inf')


class TestpdistAPICase7(TestpdistAPI):
    def init_input(self):
        self.x = np.random.rand(50, 20).astype('float64')
        self.compute_mode = "use_mm_for_euclid_dist"


class TestpdistAPICase8(TestpdistAPI):
    def init_input(self):
        self.x = np.random.rand(50, 20).astype('float64')
        self.compute_mode = "donot_use_mm_for_euclid_dist"


class TestpdistAPICase9(TestpdistAPI):
    def init_input(self):
        self.x = np.random.rand(500, 100).astype('float64')


    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            out0 = paddle.pdist(x, self.p, self.compute_mode)
            out1 = paddle.pdist(x, self.p, "donot_use_mm_for_euclid_dist")
            out2 = paddle.pdist(x, self.p, "use_mm_for_euclid_dist")
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.x}, fetch_list=[out0, out1, out2]
            )
            out_ref = ref_pdist(self.x, self.p)
            np.testing.assert_allclose(out_ref, res[0])
            np.testing.assert_allclose(out_ref, res[1])
            np.testing.assert_allclose(out_ref, res[2])

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out0 = paddle.pdist(x, self.p, self.compute_mode)
        out1 = paddle.pdist(x, self.p, "donot_use_mm_for_euclid_dist")
        out2 = paddle.pdist(x, self.p, "use_mm_for_euclid_dist")
        out_ref = ref_pdist(self.x, self.p)
        np.testing.assert_allclose(out_ref, out0.numpy())
        np.testing.assert_allclose(out_ref, out1.numpy())
        np.testing.assert_allclose(out_ref, out2.numpy())
        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()




