import paddle
import unittest
import numpy as np

class TestIInfoAndFInfoAPI(unittest.TestCase):
    def test_invalid_input(self):
        for dtype in [paddle.float16, paddle.float32, paddle.float64, paddle.bfloat16, paddle.complex64, paddle.complex128, paddle.bool]:
            # I think it's best to raise TypeError, not  ValueError
            with self.assertRaises(ValueError):
                _ = paddle.iinfo(dtype)

    def test_iinfo(self):
        for dtype in [paddle.int64, paddle.int32, paddle.int16, paddle.int8, paddle.uint8]:
            x = paddle.to_tensor([2, 3], dtype=dtype)
            xinfo = paddle.iinfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.iinfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.dtype, xninfo.dtype)

if __name__ == '__main__':
    unittest.main()
