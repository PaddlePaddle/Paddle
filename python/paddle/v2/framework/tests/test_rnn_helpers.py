import unittest
from paddle.v2.framework.layers import *
from paddle.v2.framework.framework import g_program, g_init_program
from paddle.v2.framework.executor import Executor
import numpy as np
import paddle.v2.framework.core as core


class TestRNN(unittest.TestCase):
    def test_rnn(self):

        batch_size, seq_len, input_dim, hidden_dim = 2, 3, 4, 4
        video = data(
            shape=[seq_len, input_dim],  # image width
            data_type='float32',
            name='image')

        print 'f' * 10
        rnn = StaticRNN()
        with rnn.step():
            memory = rnn.memory(
                ref=video,
                shape=(-1, hidden_dim),
                dtype='float32',
                init_value=0.0)

            img = rnn.step_input(video)
            self.assertEqual((-1, input_dim), img.shape)

            rnn_out = fc(input=img, size=hidden_dim, act='sigmoid')
            self.assertEqual((-1, hidden_dim), rnn_out.shape)

            rnn.update_memory(memory, rnn_out)

            rnn.output(rnn_out)

        out = rnn()
        print g_init_program
        # print g_program

        # import pdb
        # pdb.set_trace()

        self.assertEqual((-1, seq_len, hidden_dim), out.shape)
        place = core.CPUPlace()

        tensor = core.LoDTensor()
        tensor.set(
            np.random.random(
                (batch_size, seq_len, hidden_dim)).astype("float32"), place)

        exe = Executor(place)
        outs = exe.run(g_init_program, {}, {})
        # print '*' * 20
        outs = exe.run(g_program, feed={video.name: tensor}, fetch_list=[out])


if __name__ == '__main__':
    unittest.main()
