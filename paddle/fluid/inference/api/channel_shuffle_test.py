import unittest
import paddle
from paddle.api.h import channel_shuffle
class ChannelShuffleTest(unittest.TestCase):
    def test_identity_matrix(self):
        input = paddle.to_tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3, 1, 1])
        groups = 1
        output = paddle.zeros([2, 3, 1, 1])

        channel_shuffle(input, groups, output)

        expected_output = paddle.to_tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3, 1, 1])
        self.assertTrue(paddle.allclose(expected_output, output))

if __name__ == '__main__':
    unittest.main()
