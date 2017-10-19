import unittest
from paddle.v2.framework.layers import mul, data_layer
import paddle.v2.framework.core as core
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.framework import g_program
import numpy


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        a = data_layer(name='a', shape=[784], data_type='float32')
        b = data_layer(
            name='b',
            shape=[784, 100],
            data_type='float32',
            append_batch_size=False)
        out = mul(x=a, y=b)
        place = core.CPUPlace()
        a_np = numpy.random.random((100, 784)).astype('float32')
        tensor_a = core.LoDTensor()
        tensor_a.set(a_np, place)
        b_np = numpy.random.random((784, 100)).astype('float32')
        tensor_b = core.LoDTensor()
        tensor_b.set(b_np, place)
        # del input_tensor
        exe = Executor(place)
        outs = list(
            exe.run(g_program,
                    feed={'a': tensor_a,
                          'b': tensor_b},
                    fetch_list=[out]))
        out = numpy.array(outs[0])
        self.assertEqual((100, 100), out.shape)


if __name__ == '__main__':
    unittest.main()
