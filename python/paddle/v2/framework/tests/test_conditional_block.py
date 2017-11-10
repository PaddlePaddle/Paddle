import unittest
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
from paddle.v2.framework.framework import g_startup_program, g_main_program
from paddle.v2.framework.executor import Executor
import numpy


class ConditionalBlock(unittest.TestCase):
    def test_forward(self):
        data = layers.data(name='X', shape=[1], data_type='float32')
        cond = layers.ConditionalBlock(inputs=[data])
        out = layers.create_tensor(dtype='float32')
        with cond.block():
            hidden = layers.fc(input=data, size=10)
            layers.assign(hidden, out)

        cpu = core.CPUPlace()
        exe = Executor(cpu)
        exe.run(g_startup_program)

        x = core.LoDTensor()
        x.set(numpy.random.random(size=(10, 1)).astype('float32'), cpu)

        outs = map(numpy.array, exe.run(feed={'X': x}, fetch_list=[out]))[0]


if __name__ == '__main__':
    unittest.main()
