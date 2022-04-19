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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard

np.random.seed(10)


class TestNanmeanAPI(unittest.TestCase):
    # test paddle.tensor.math.nanmean

    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.x[0, :, :, :] = np.nan
        self.x_grad = np.array([[np.nan, np.nan, 3.],
                                [0., np.nan, 2.]]).astype(np.float32)
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_shape)
            out1 = paddle.nanmean(x)
            out2 = paddle.tensor.nanmean(x)
            out3 = paddle.tensor.math.nanmean(x)
            axis = np.arange(len(self.x_shape)).tolist()
            out4 = paddle.nanmean(x, axis)
            out5 = paddle.nanmean(x, tuple(axis))
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x},
                          fetch_list=[out1, out2, out3, out4, out5])
        out_ref = np.nanmean(self.x)
        for out in res:
            self.assertEqual(np.allclose(out, out_ref, rtol=1e-04), True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        def test_case(x, axis=None, keepdim=False):
            x_tensor = paddle.to_tensor(x)
            out = paddle.nanmean(x_tensor, axis, keepdim)
            if isinstance(axis, list):
                axis = tuple(axis)
                if len(axis) == 0:
                    axis = None

            out_ref = np.nanmean(x, axis, keepdims=keepdim)
            if np.isnan(out_ref).sum():
                nan_mask = np.isnan(out_ref)
                out_ref[nan_mask] = 0
                out_np = out.numpy()
                out_np[nan_mask] = 0
                self.assertEqual(np.allclose(out_np, out_ref, rtol=1e-04), True)
            else:
                self.assertEqual(
                    np.allclose(
                        out.numpy(), out_ref, rtol=1e-04), True)

        test_case(self.x)
        test_case(self.x, [])
        test_case(self.x, -1)
        test_case(self.x, keepdim=True)
        test_case(self.x, 2, keepdim=True)
        test_case(self.x, [0, 2])
        test_case(self.x, (0, 2))
        test_case(self.x, [0, 1, 2, 3])
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12], 'int32')
            self.assertRaises(TypeError, paddle.nanmean, x)

    def test_api_dygraph_grad(self):
        paddle.disable_static(self.place)

        def test_case(x, axis=None, keepdim=False):
            if isinstance(axis, list):
                axis = list(axis)
                if len(axis) == 0:
                    axis = None
            x_tensor = paddle.to_tensor(x, stop_gradient=False)
            y = paddle.nanmean(x_tensor, axis, keepdim)
            dx = paddle.grad(y, x_tensor)[0].numpy()
            sum_dx_ref = np.prod(y.shape)
            if np.isnan(y.numpy()).sum():
                sum_dx_ref -= np.isnan(y.numpy()).sum()
            cnt = paddle.sum(~paddle.isnan(x_tensor),
                             axis=axis,
                             keepdim=keepdim)
            if (cnt == 0).sum():
                dx[np.isnan(dx)] = 0
            sum_dx = dx.sum()
            self.assertEqual(np.allclose(sum_dx, sum_dx_ref, rtol=1e-04), True)

        test_case(self.x)
        test_case(self.x, [])
        test_case(self.x, -1)
        test_case(self.x, keepdim=True)
        test_case(self.x, 2, keepdim=True)
        test_case(self.x, [0, 2])
        test_case(self.x, (0, 2))
        test_case(self.x, [0, 1, 2, 3])

        test_case(self.x_grad)
        test_case(self.x_grad, [])
        test_case(self.x_grad, -1)
        test_case(self.x_grad, keepdim=True)
        test_case(self.x_grad, 0, keepdim=True)
        test_case(self.x_grad, 1)
        test_case(self.x_grad, (0, 1))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
