import unittest
from paddle.v2.framework.layers import mul, data, sequence_pool
import paddle.v2.framework.core as core
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.framework import g_main_program
import numpy


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        a = data(name='a', shape=[784], data_type='float32')
        b = data(
            name='b',
            shape=[784, 100],
            data_type='float32',
            append_batch_size=False)
        out = mul(x=a, y=b)
        place = core.CPUPlace()
        a_np = numpy.random.random((100, 784)).astype('float32')
        b_np = numpy.random.random((784, 100)).astype('float32')
        exe = Executor(place)
        outs = exe.run(g_main_program,
                       feed={'a': a_np,
                             'b': b_np},
                       fetch_list=[out])
        out = outs[0]
        self.assertEqual((100, 100), out.shape)
        self.assertTrue(numpy.allclose(out, numpy.dot(a_np, b_np)))


def sequence_sum(lod, x):
    N = len(lod[0]) - 1
    out = [None for i in range(N)]
    for i in range(N):
        sub_x = x[lod[0][i]:lod[0][i + 1], :]
        out[i] = sub_x.sum(axis=0)
    return out


class TestExecutor2(unittest.TestCase):
    def test_asnumpy(self):
        seq = data(name='seq', shape=[784], data_type='float32')
        out = sequence_pool(seq, "sum")
        x = np.ones(shape=(3, 5)).astype('float32')
        lod = [[0, 2, 3]]
        tensor = core.LoDTensor()
        place = core.CPUPlace()
        tensor.set(x, place)
        tensor.set_lod(lod)
        exe = Executor(place)
        outs = exe.run(g_main_program, feed={"seq": tensor}, fetch_list=out)
        self.assertTrue(np.allclose(outs[0], sequence_sum(lod, x)))


if __name__ == '__main__':
    unittest.main()
