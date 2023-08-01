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
from paddle import fluid
from paddle.fluid import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)
class TestRMSNormOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        batch = 32
        cols = 256
        self.x_np = np.random.random([batch, 256])
        self.gamma_np = np.random.random([256])
        self.beta_np = np.random.random([256])
        self.epsilon = 1e-6

    def naive_rms_norm(self, x, gamma, beta):
        variance = x.pow(2).mean(-1, keepdim=True)
        out = paddle.rsqrt(variance + self.epsilon) * x
        out = out * gamma + beta
        return out

    def check_main(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_rmsnorm_out = paddle.incubate.nn.functional.rms_norm(
            x, gamma, beta, self.epsilon, begin_norm_axis=1
        )
        paddle_naive_rmsnorm_out = self.naive_rms_norm(x, gamma, beta)
        paddle.enable_static()
        return paddle_rmsnorm_out, paddle_naive_rmsnorm_out

    def test_rmsnorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_main(
            self.x_np, self.gamma_np, self.beta_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm.numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-03,
            atol=1e-3,
        )

    def test_rmsnorm_fp32(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_main(
            self.x_np, self.gamma_np, self.beta_np, 'float32'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm.numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)
class TestRMSNormStaticOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.batch = 32
        self.cols = 256
        self.x_np = np.random.random([self.batch, 256])
        self.gamma_np = np.random.random([256])
        self.beta_np = np.random.random([256])
        self.epsilon = 1e-6
        self.place = paddle.CUDAPlace(0)

    def naive_rms_norm(self, x, gamma, beta):
        variance = x.pow(2).mean(-1, keepdim=True)
        out = paddle.rsqrt(variance + self.epsilon) * x
        out = out * gamma + beta
        return out

    def check_main(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_naive_rmsnorm_out = self.naive_rms_norm(x, gamma, beta)
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )

            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype=dtype
            )

            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype=dtype
            )

            outs = paddle.incubate.nn.functional.rms_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
            )

            exe = fluid.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(dtype),
                    "beta_static": beta_np.astype(dtype),
                },
                fetch_list=[outs],
            )

        return out_s[0], paddle_naive_rmsnorm_out

    def test_rmsnorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_main(
            self.x_np, self.gamma_np, self.beta_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm,
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_rmsnorm_fp32(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_main(
            self.x_np, self.gamma_np, self.beta_np, 'float32'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm,
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
