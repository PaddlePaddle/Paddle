import unittest
from paddle.v2.framework.layers import *
from paddle.v2.framework.framework import g_program


class TestRNN(unittest.TestCase):
    def test_rnn(self):
        img = data(
            shape=[
                80,  # sequence length
                -1,  # batch size
                22,  # image height
                22
            ],  # image width
            data_type='float32',
            name='image')
        hidden = fc(input=img, size=100, act='sigmoid', num_flatten_dims=2)
        self.assertEqual((80, -1, 100), hidden.shape)
        hidden = fc(input=hidden, size=100, act='sigmoid', num_flatten_dims=2)
        self.assertEqual((80, -1, 100), hidden.shape)

        rnn = StaticRNN()
        with rnn.step():
            hidden = rnn.step_input(hidden)
            self.assertEqual((-1, 100), hidden.shape)
            memory = rnn.memory(shape=(-1, 32), dtype='float32', init_value=0.0)

            rnn_out = fc(input=[hidden, memory], size=32, act='sigmoid')
            self.assertEqual((-1, 32), rnn_out.shape)
            rnn.update_memory(memory, rnn_out)
            rnn.output(rnn_out)

        out = rnn()
        self.assertEqual((80, -1, 32), out.shape)
        print g_program


if __name__ == '__main__':
    unittest.main()
