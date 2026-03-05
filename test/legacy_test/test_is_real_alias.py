import unittest
import paddle
import numpy as np

class TestIsRealAlias(unittest.TestCase):
    def test_input_alias(self):
        # 准备数据
        data = np.array([1.0, 2.0 + 1j, 3.0], dtype='complex64')
        x = paddle.to_tensor(data)
        
        # 1. 验证原始参数名 x (PHI 算子下沉后的标准参数)
        out_x = paddle.is_real(x=x)
        
        # 2. 验证新增的别名参数名 input (为了兼容 PyTorch)
        # 如果你 YAML 配置正确，这里不会报错
        out_input = paddle.is_real(input=x)
        
        # 3. 验证结果是否一致
        expected = np.isreal(data)
        self.assertTrue(np.allclose(out_x.numpy(), expected))
        self.assertTrue(np.allclose(out_input.numpy(), expected))

    def test_tensor_method(self):
        # 验证 Tensor 形式的调用：x.is_real()
        x = paddle.to_tensor([1.0, 2.0 + 1j])
        self.assertTrue(x.is_real().numpy()[0])
        self.assertFalse(x.is_real().numpy()[1])

if __name__ == '__main__':
    unittest.main()