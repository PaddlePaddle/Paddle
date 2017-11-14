import unittest
import paddle.v2.framework.core as core
from paddle.v2.framework.executor import Executor
import paddle.v2.framework.layers as layers
from paddle.v2.framework.backward import append_backward_ops
from paddle.v2.framework.framework import g_main_program
import numpy


class TestShrinkRNNMemory(unittest.TestCase):
    def test_shrink_rnn_memory(self):
        x = layers.data('x', shape=[100], data_type='float32')
        x.stop_gradient = False
        table = layers.lod_rank_table(x=x)
        i = layers.zeros(dtype='int64', shape=[1])
        mem1 = layers.shrink_memory(x=x, i=i, table=table)
        i = layers.increment(x=i)
        i.stop_gradient = True
        mem2 = layers.shrink_memory(x=mem1, i=i, table=table)
        i = layers.increment(x=i)
        i.stop_gradient = True
        mem3 = layers.shrink_memory(x=mem2, i=i, table=table)

        cpu = core.CPUPlace()
        tensor = core.LoDTensor()
        tensor.set_lod([[0, 2, 5, 6]])
        tensor_np = numpy.random.random(size=(3, 100)).astype('float32')
        tensor.set(tensor_np, cpu)
        exe = Executor(cpu)
        outs = map(numpy.array,
                   exe.run(feed={'x': tensor}, fetch_list=[mem1, mem2, mem3]))
        self.assertTrue(numpy.allclose(tensor_np[0:3], outs[0]))
        self.assertTrue(numpy.allclose(tensor_np[0:2], outs[1]))
        self.assertTrue(numpy.allclose(tensor_np[0:1], outs[2]))

        mem3_mean = layers.mean(x=mem3)
        append_backward_ops(loss=mem3_mean)
        x_grad = map(numpy.array,
                     exe.run(feed={'x': tensor},
                             fetch_list=[
                                 g_main_program.global_block().var('x@GRAD')
                             ]))[0]
        self.assertAlmostEqual(1.0, x_grad.sum(), delta=0.1)


if __name__ == '__main__':
    unittest.main()
