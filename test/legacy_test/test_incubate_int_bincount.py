import paddle
import numpy as np
import unittest
from paddle.incubate.nn.functional import int_bincount

class TestIntBincount(unittest.TestCase):
    def setUp(self):
        paddle.set_device('gpu')
    
    def test_basic(self):
        x = paddle.to_tensor([1, 2, 3, 1, 2, 3], dtype=paddle.int32)
        out = int_bincount(x, low=1, high=4, dtype=paddle.int32)
        expected = np.array([2, 2, 2,0])
        np.testing.assert_array_equal(out.numpy(), expected)
    
    def test_empty_input(self):
        x = paddle.to_tensor([], dtype=paddle.int32)
        out = int_bincount(x, low=0, high=10, dtype=paddle.int32)
        self.assertEqual(out.shape, [11])
        self.assertEqual(out.sum().item(), 0)
    
    def test_different_dtypes(self):
        x = paddle.to_tensor([1, 3, 5, 3, 1], dtype=paddle.int64)
        out = int_bincount(x, low=1, high=6, dtype=paddle.int64)
        expected = np.array([2, 0, 2, 0, 1, 0])
        np.testing.assert_array_equal(out.numpy(), expected)
    

if __name__ == '__main__':
    unittest.main()