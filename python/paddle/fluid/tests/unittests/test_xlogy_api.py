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
from scipy import special

import paddle
import paddle.fluid.core as core

np.random.seed(10)


def ref_xlogy(x, other):
    out = special.xlogy(x, other)
    return out


class TestXlogyAPI(unittest.TestCase):
    # test paddle.tensor.math.xlogy

    def setUp(self, type=None):
        self.shape = [2, 3, 3, 5]
        self.x = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        self.other = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        if type == 'nan_test':
            self.x[:, 0, :, :] = 0.0
            self.x[:, 1, :, :] = float('inf')
            self.x[:, 2, :, :] = float('nan')
            self.other[:, :, 0, :] = 0.0
            self.other[:, :, 1, :] = float('inf')
            self.other[:, :, 2, :] = float('nan')
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def api_case(self, type=None):
        self.setUp(type)
        out_ref = ref_xlogy(self.x, self.other)
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape)
            other = paddle.fluid.data('Other', self.shape)
            out = paddle.xlogy(x, other)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'X': self.x, 'Other': self.other}, fetch_list=[out]
            )
        ref_nan_mask = np.isnan(out_ref)
        res_nan_mask = np.isnan(res[0])
        np.testing.assert_allclose(ref_nan_mask, res_nan_mask)
        np.testing.assert_allclose(
            res[0][~res_nan_mask], out_ref[~ref_nan_mask], rtol=1e-05
        )

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        other = paddle.to_tensor(self.other)
        res = paddle.xlogy(x, other)
        ref_nan_mask = np.isnan(out_ref)
        res_nan_mask = paddle.isnan(res).numpy()
        np.testing.assert_allclose(ref_nan_mask, res_nan_mask)
        np.testing.assert_allclose(
            res[~res_nan_mask], out_ref[~ref_nan_mask], rtol=1e-05
        )
        paddle.enable_static()

    def test_api_dygraph_grad(self, type=None):
        self.setUp(type)
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        other = paddle.to_tensor(self.other, stop_gradient=False)
        res = paddle.xlogy(x, other)
        res.sum().backward()
        mask = (x == 0) & ((other <= 0) | (other == float('inf')))
        tmp_other = paddle.where(
            mask, paddle.ones(other.shape, other.dtype), other
        )
        ref_x_grad = paddle.log(tmp_other)
        ref_other_grad = x / tmp_other
        for grads in [(ref_x_grad, x.grad), (ref_other_grad, other.grad)]:
            ref_nan_mask = paddle.isnan(grads[0])
            res_nan_mask = paddle.isnan(grads[1])
            np.testing.assert_allclose(ref_nan_mask, res_nan_mask)
            np.testing.assert_allclose(
                grads[1][~res_nan_mask].numpy(),
                grads[0][~ref_nan_mask].numpy(),
                rtol=1e-05,
            )
        paddle.enable_static()

    def test_api(self):
        self.api_case('nan_test')
        self.api_case()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12], 'int32')
            self.assertRaises(TypeError, paddle.xlogy, x)

    def test_alias(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        other = paddle.to_tensor(self.other)
        out1 = paddle.xlogy(x, other)
        out2 = paddle.tensor.xlogy(x, other)
        out3 = paddle.tensor.math.xlogy(x, other)
        out_ref = ref_xlogy(self.x, self.other)
        for out in [out1, out2, out3]:
            ref_nan_mask = np.isnan(out_ref)
            res_nan_mask = paddle.isnan(out)
            np.testing.assert_allclose(ref_nan_mask, res_nan_mask)
            np.testing.assert_allclose(
                out[~res_nan_mask], out_ref[~ref_nan_mask], rtol=1e-05
            )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
