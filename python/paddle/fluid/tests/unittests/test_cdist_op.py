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
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        paddle.enable_static()

        # case 1
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(10, 20).astype('float32')
            y_np = np.random.rand(11, 20).astype('float32')
            p = 2.0
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 2
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(10, 20).astype('float32')
            y_np = np.random.rand(11, 20).astype('float32')
            p = 0
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 3
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(10, 20).astype('float32')
            y_np = np.random.rand(11, 20).astype('float32')
            p = 1.0
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 4
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(10, 20).astype('float32')
            y_np = np.random.rand(11, 20).astype('float32')
            p = 3.0
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 5
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(10, 20).astype('float32')
            y_np = np.random.rand(11, 20).astype('float32')
            p = 1.5
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 6
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(10, 20).astype('float32')
            y_np = np.random.rand(11, 20).astype('float32')
            p = 2.5
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 7
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(10, 20).astype('float32')
            y_np = np.random.rand(11, 20).astype('float32')
            p = 2
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p, 'use_mm_for_euclid_dist')
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 8
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(50, 10).astype('float32')
            y_np = np.random.rand(40, 10).astype('float32')
            p = 2
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p, 'donot_use_mm_for_euclid_dist')
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 9
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(500, 100).astype('float64')
            y_np = np.random.rand(400, 100).astype('float64')
            p = 2
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 10
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(4, 500, 100).astype('float64')
            y_np = np.random.rand(4, 400, 100).astype('float64')
            p = 2
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 11
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(3, 4, 500, 100).astype('float64')
            y_np = np.random.rand(3, 4, 400, 100).astype('float64')
            p = 2
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

        # case 12
        with paddle.static.program_guard(paddle.static.Program()):
            x_np = np.random.rand(3, 4, 500, 100).astype('float64')
            y_np = np.random.rand(3, 4, 400, 100).astype('float64')
            p = 3
            x = paddle.static.data('x', x_np.shape, dtype=x_np.dtype)
            y = paddle.static.data('y', y_np.shape, dtype=y_np.dtype)
            out = paddle.cdist(x, y, p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x_np, 'y': y_np}, fetch_list=[out])
            out_ref = ref_cdist(x_np, y_np, p)
            self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)

        # case 1
        x_np = np.random.rand(10, 20).astype('float32')
        y_np = np.random.rand(11, 20).astype('float32')
        p = 2
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 2
        x_np = np.random.rand(10, 20).astype('float32')
        y_np = np.random.rand(11, 20).astype('float32')
        p = 0
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 3
        x_np = np.random.rand(10, 20).astype('float32')
        y_np = np.random.rand(11, 20).astype('float32')
        p = 1.5
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 4
        x_np = np.random.rand(10, 20).astype('float32')
        y_np = np.random.rand(11, 20).astype('float32')
        p = float('inf')
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 5
        x_np = np.random.rand(10, 20).astype('float32')
        y_np = np.random.rand(11, 20).astype('float32')
        p = 2
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p, 'use_mm_for_euclid_dist')
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 6
        x_np = np.random.rand(10, 20).astype('float32')
        y_np = np.random.rand(11, 20).astype('float32')
        p = 2
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p, 'donot_use_mm_for_euclid_dist')
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 7
        x_np = np.random.rand(100, 200).astype('float64')
        y_np = np.random.rand(110, 200).astype('float64')
        p = 2
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 7
        x_np = np.random.rand(100, 200).astype('float64')
        y_np = np.random.rand(110, 200).astype('float64')
        p = 3
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 8
        x_np = np.random.rand(3, 100, 200).astype('float64')
        y_np = np.random.rand(3, 110, 200).astype('float64')
        p = 2
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        # case 9
        x_np = np.random.rand(2, 3, 100, 200).astype('float64')
        y_np = np.random.rand(2, 3, 110, 200).astype('float64')
        p = 3
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.cdist(x, y, p)
        out_ref = ref_cdist(x_np, y_np, p)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
