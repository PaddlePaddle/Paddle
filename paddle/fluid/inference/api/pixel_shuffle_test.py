import unittest
import paddle
from paddle.api.h import pixel_shuffle
class PixelShuffleTest(unittest.TestCase):
    def test_identity_matrix(self):
        input = paddle.to_tensor([1, 2, 3, 4]).reshape([1, 2, 2])
        upscale_factor = 2
        output = paddle.zeros([4, 1, 1])

        pixel_shuffle(input, upscale_factor, output)

        expected_output = paddle.to_tensor([1, 2, 3, 4]).reshape([4, 1, 1])
        self.assertTrue(paddle.allclose(expected_output, output))

if __name__ == '__main__':
    unittest.main()
