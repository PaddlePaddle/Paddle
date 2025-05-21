import unittest
import numpy as np
import paddle
from paddle.fluid import core

# 导入反向算子
from paddle.incubate.nn.functional.fused_rms_norm_grad import fused_rms_norm_grad


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFusedRMSNormGradOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.batch = 4
        self.cols = 8
        self.x_np = np.random.random([self.batch, self.cols]).astype("float32")
        self.gamma_np = np.random.random([self.cols]).astype("float32")
        self.epsilon = 1e-6
        # 随机上游梯度
        self.dy_np = np.random.random([self.batch, self.cols]).astype("float32")
        # 数值梯度扰动
        self.delta = 1e-3

    def naive_forward(self, x, gamma, epsilon):
        var = np.mean(x * x, axis=-1, keepdims=True)
        out = x / np.sqrt(var + epsilon)
        return out * gamma

    def numeric_gradients(self):
        def loss(x, gamma):
            out = self.naive_forward(x, gamma, self.epsilon)
            return np.sum(out * self.dy_np)

        grad_x_num = np.zeros_like(self.x_np)
        for i in range(self.batch):
            for j in range(self.cols):
                x_plus = self.x_np.copy()
                x_minus = self.x_np.copy()
                x_plus[i, j] += self.delta
                x_minus[i, j] -= self.delta
                l_plus = loss(x_plus, self.gamma_np)
                l_minus = loss(x_minus, self.gamma_np)
                grad_x_num[i, j] = (l_plus - l_minus) / (2 * self.delta)

        grad_gamma_num = np.zeros_like(self.gamma_np)
        for j in range(self.cols):
            gamma_plus = self.gamma_np.copy()
            gamma_minus = self.gamma_np.copy()
            gamma_plus[j] += self.delta
            gamma_minus[j] -= self.delta
            l_plus = loss(self.x_np, gamma_plus)
            l_minus = loss(self.x_np, gamma_minus)
            grad_gamma_num[j] = (l_plus - l_minus) / (2 * self.delta)

        return grad_x_num, grad_gamma_num

    def test_fused_rms_norm_grad_fp32(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        gamma = paddle.to_tensor(self.gamma_np, stop_gradient=False)

        # 随机生成 invvar 和 dy，不通过前向算子计算
        var = np.mean(self.x_np * self.x_np, axis=-1)
        invvar_np = 1.0 / np.sqrt(var + self.epsilon)
        invvar = paddle.to_tensor(invvar_np, dtype="float32")
        dy = paddle.to_tensor(self.dy_np, dtype="float32")

        grad_x, grad_gamma = fused_rms_norm_grad(
            x, gamma, invvar, dy, epsilon=self.epsilon
        )

        grad_x_np = grad_x.numpy()
        grad_gamma_np = grad_gamma.numpy()

        grad_x_num, grad_gamma_num = self.numeric_gradients()

        np.testing.assert_allclose(
            grad_x_np, grad_x_num, rtol=1e-2, atol=1e-3
        )
        np.testing.assert_allclose(
            grad_gamma_np, grad_gamma_num, rtol=1e-2, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
