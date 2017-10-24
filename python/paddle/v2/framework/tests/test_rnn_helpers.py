import unittest
from paddle.v2.framework.layers import *
from paddle.v2.framework.framework import g_program, g_init_program
from paddle.v2.framework.executor import Executor
import numpy as np
import paddle.v2.framework.core as core


class TestRNN(unittest.TestCase):
    def test_rnn(self):
        batch_size, seq_len, height, weight = 2, 80, 22, 22
        img = data(
            shape=[seq_len, height, weight],  # image width
            data_type='float32',
            name='image')
        hidden = fc(input=img, size=100, act='sigmoid', num_flatten_dims=2)
        self.assertEqual((-1, 80, 100), hidden.shape)
        # hidden = fc(input=hidden, size=100, act='sigmoid', num_flatten_dims=2)
        # self.assertEqual((-1, 80, 100), hidden.shape)

        rnn = StaticRNN()
        with rnn.step():
            hidden = rnn.step_input(hidden)
            self.assertEqual((-1, 100), hidden.shape)
            memory = rnn.memory(
                ref=img, shape=(-1, 32), dtype='float32', init_value=0.0)

            rnn_out = fc(input=[hidden, memory], size=32, act='sigmoid')
            self.assertEqual((-1, 32), rnn_out.shape)
            rnn.update_memory(memory, rnn_out)
            rnn.output(rnn_out)

        out = rnn()
        print g_program

        self.assertEqual((-1, 80, 32), out.shape)
        place = core.CPUPlace()

        tensor = core.LoDTensor()
        tensor.set(
            np.random.random(
                (batch_size, seq_len, height, weight)).astype("float32"), place)

        exe = Executor(place)
        outs = exe.run(g_init_program, {}, {})
        print '*' * 20
        outs = exe.run(g_program, feed={img.name: tensor}, fetch_list=[])


if __name__ == '__main__':
    unittest.main()
