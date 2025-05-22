import unittest
import paddle
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)

class TestIntBincount(unittest.TestCase):
    def test_basic(self):
        # 输入值 0…5，各出现次数分别为 [1,1,1,3,0,1]
        x = paddle.to_tensor([0, 2, 1, 2, 5, 2], dtype='int64')
        # low=0, high=5, 默认 dtype 与 x 相同
        y = paddle.nn.functional.int_bincount(x, low=0, high=5)
        expected = np.array([1, 1, 1, 3, 0, 1], dtype='int64')
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_with_dtype(self):
        # 指定输出 dtype 为 int32
        x = paddle.to_tensor([1, 3, 3, 1, 0], dtype='int32')
        # low=0, high=3
        y = paddle.nn.functional.int_bincount(x, low=0, high=3, dtype='int32')
        # 值 0→1 次, 1→2 次, 2→0 次, 3→2 次
        expected = np.array([1, 2, 0, 2], dtype='int32')
        self.assertEqual(y.dtype, paddle.int32)
        np.testing.assert_array_equal(y.numpy(), expected)

if __name__ == '__main__':
    unittest.main()