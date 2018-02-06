import unittest

import numpy
import paddle.v2.fluid.core as core

from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.layers import mul, data


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        a = data(name='a', shape=[784], dtype='float32')
        b = data(
            name='b',
            shape=[784, 100],
            dtype='float32',
            append_batch_size=False)
        out = mul(x=a, y=b)
        place = core.CPUPlace()
        a_np = numpy.random.random((100, 784)).astype('float32')
        b_np = numpy.random.random((784, 100)).astype('float32')
        exe = Executor(place)
        outs = exe.run(feed={'a': a_np, 'b': b_np}, fetch_list=[out])
        out = outs[0]
        self.assertEqual((100, 100), out.shape)
        self.assertTrue(numpy.allclose(out, numpy.dot(a_np, b_np)))


if __name__ == '__main__':
    unittest.main()
