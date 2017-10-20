import unittest
from paddle.v2.framework.layer_helper import StaticRNNHelper
from paddle.v2.framework.layers import *


class TestRNN(unittest.TestCase):
    def test_rnn(self):
        img = data(
            shape=[
                80,  # sequence length
                22,  # image height
                22
            ],  # image width
            data_type='float32',
            name='image')
        print img.shape
        hidden = fc(input=img, size=100, act='sigmoid', num_flatten_dims=2)
        print hidden.shape  # [-1, 80, 100]
        hidden = fc(input=hidden, size=100, act='sigmoid', num_flatten_dims=2)


if __name__ == '__main__':
    unittest.main()
