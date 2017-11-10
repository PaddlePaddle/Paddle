import unittest
import paddle.v2.framework.layers as layers
from paddle.v2.framework.executor import Executor
import paddle.v2.framework.core as core
import numpy


class TestLoDArrayLength(unittest.TestCase):
    def test_array_length(self):
        tmp = layers.zeros(shape=[10], dtype='int32')
        i = layers.fill_constant(shape=[1], dtype='int64', value=10)
        arr = layers.array_write(tmp, i=i)
        arr_len = layers.array_length(arr)
        cpu = core.CPUPlace()
        exe = Executor(cpu)
        result = numpy.array(exe.run(fetch_list=[arr_len])[0])
        self.assertEqual(11, result[0])


if __name__ == '__main__':
    unittest.main()
