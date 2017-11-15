import unittest
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.core as core
from paddle.v2.fluid.framework import g_startup_program, g_main_program
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.backward import append_backward_ops
import numpy


class ConditionalBlock(unittest.TestCase):
    def test_forward(self):
        data = layers.data(name='X', shape=[1], data_type='float32')
        data.stop_gradient = False
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
        print outs
        loss = layers.mean(x=out)
        append_backward_ops(loss=loss)
        outs = map(numpy.array,
                   exe.run(feed={'X': x},
                           fetch_list=[
                               g_main_program.block(0).var(data.name + "@GRAD")
                           ]))[0]
        print outs


if __name__ == '__main__':
    unittest.main()
