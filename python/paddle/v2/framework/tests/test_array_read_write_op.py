import unittest

import numpy
import paddle.v2.framework.core as core

import paddle.v2.framework.layers as layers
from paddle.v2.framework.executor import Executor


class TestArrayReadWrite(unittest.TestCase):
    def test_read_write(self):
        x = [
            layers.data(
                name='x0', shape=[100]), layers.data(
                    name='x1', shape=[100]), layers.data(
                        name='x2', shape=[100])
        ]

        for each_x in x:
            each_x.stop_gradient = False

        i = layers.zeros(shape=[1], dtype='int64')
        arr = layers.array_write(x=x[0], i=i)
        layers.increment(x=i)
        arr = layers.array_write(x=x[1], i=i, array=arr)
        layers.increment(x=i)
        arr = layers.array_write(x=x[2], i=i, array=arr)

        i = layers.zeros(shape=[1], dtype='int64')
        a0 = layers.array_read(array=arr, i=i)
        layers.increment(x=i)
        a1 = layers.array_read(array=arr, i=i)
        layers.increment(x=i)
        a2 = layers.array_read(array=arr, i=i)

        mean_a0 = layers.mean(x=a0)
        mean_a1 = layers.mean(x=a1)
        mean_a2 = layers.mean(x=a2)

        a_sum = layers.sums(input=[mean_a0, mean_a1, mean_a2])

        mean_x0 = layers.mean(x=x[0])
        mean_x1 = layers.mean(x=x[1])
        mean_x2 = layers.mean(x=x[2])

        x_sum = layers.sums(input=[mean_x0, mean_x1, mean_x2])

        scope = core.Scope()
        cpu = core.CPUPlace()

        exe = Executor(cpu)

        tensor = core.LoDTensor()
        tensor.set(numpy.random.random(size=(100, 100)).astype('float32'), cpu)

        outs = map(numpy.array,
                   exe.run(feed={'x0': tensor,
                                 'x1': tensor,
                                 'x2': tensor},
                           fetch_list=[a_sum, x_sum],
                           scope=scope))
        self.assertEqual(outs[0], outs[1])


if __name__ == '__main__':
    unittest.main()
