# File: test/legacy_test/test_cuda_rms_norm_op.py

import unittest
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.executor import Executor

from paddle.incubate.nn.functional.fused_rms_norm_ext import fused_rms_norm_ext


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFusedRMSNormOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.batch = 32
        self.cols = 256
        self.x_np = np.random.random([self.batch, self.cols]).astype("float32")
        self.gamma_np = np.random.random([self.cols]).astype("float32")
        self.epsilon = 1e-6

    def naive_rms_norm(self, x, gamma, epsilon):
        var = np.mean(x * x, axis=-1, keepdims=True)
        out = x / np.sqrt(var + epsilon)
        return out * gamma

    def test_fused_rms_norm_fp16(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np.astype("float16"))
        gamma = paddle.to_tensor(self.gamma_np.astype("float16"))
        y, invvar = fused_rms_norm(x, gamma, epsilon=self.epsilon)
        y_ref = self.naive_rms_norm(self.x_np, self.gamma_np, self.epsilon)
        np.testing.assert_allclose(
            y.numpy().astype("float32"),
            y_ref,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_fused_rms_norm_fp32(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np.astype("float32"))
        gamma = paddle.to_tensor(self.gamma_np.astype("float32"))
        y, invvar = fused_rms_norm(x, gamma, epsilon=self.epsilon)
        y_ref = self.naive_rms_norm(self.x_np, self.gamma_np, self.epsilon)
        np.testing.assert_allclose(
            y.numpy(),
            y_ref,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()