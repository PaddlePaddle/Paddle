import unittest
from paddle.v2.framework.layers import *
from paddle.v2.framework.executor import Executor
import paddle.v2.framework.core as core
from paddle.v2.framework.framework import g_program, g_init_program
import numpy


class TestRNN(unittest.TestCase):
    def test_rnn(self):
        img = data(
            shape=[
                80,  # sequence length
                40,  # batch size
                22,  # image height
                22
            ],  # image width
            data_type='float32',
            name='image',
            append_batch_size=False)
        hidden = fc(input=img, size=100, act='sigmoid', num_flatten_dims=2)
        self.assertEqual((80, 40, 100), hidden.shape)
        hidden = fc(input=hidden, size=100, act='sigmoid', num_flatten_dims=2)
        self.assertEqual((80, 40, 100), hidden.shape)

        rnn = StaticRNN()
        with rnn.step():
            hidden = rnn.step_input(hidden)
            self.assertEqual((40, 100), hidden.shape)
            memory = rnn.memory(shape=(40, 32), dtype='float32', init_value=0.0)

            rnn_out = fc(input=[hidden, memory], size=32, act='sigmoid')
            self.assertEqual((40, 32), rnn_out.shape)
            rnn.update_memory(memory, rnn_out)
            rnn.output(rnn_out)

        out = rnn()
        self.assertEqual((80, 40, 32), out.shape)
        exe = Executor(core.CPUPlace())
        exe.run(g_init_program)
        tensor = core.LoDTensor()
        tensor.set(numpy.random.random(size=(80, 40, 22, 22)).astype('float32'),
                   core.CPUPlace())
        out_var = exe.run(g_program, feed={'image': tensor}, fetch_list=[out])
        print numpy.array(out_var[0])


if __name__ == '__main__':
    unittest.main()
