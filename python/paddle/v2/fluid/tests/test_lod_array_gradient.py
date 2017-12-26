import paddle.v2.fluid as fluid
import unittest
import random
import numpy


def run_many_times(times):
    def __time_fn__(fn):
        def __impl__(*args, **kwargs):
            r = list()
            for _ in xrange(times):
                r.append(fn(*args, **kwargs))
            return r

        return __impl__

    return __time_fn__


def new_program_scope(fn):
    def __impl__(*args, **kwargs):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            return fn(*args, **kwargs)

    return __impl__


def rand_lod_tensor(max_seq_len, max_num_seq, shape, place, dtype='float32'):
    num_seq = random.randint(1, max_num_seq - 1)
    lod = [0]
    for _ in xrange(num_seq):
        seq_len = random.randint(1, max_seq_len)
        lod.append(lod[-1] + seq_len)

    arr = numpy.random.random(size=[lod[-1]] + shape).astype(dtype=dtype)
    tensor = fluid.LoDTensor()
    tensor.set(arr, place)
    tensor.set_lod([lod])
    return tensor


class TestLodToArrayGradientCheck(unittest.TestCase):
    @run_many_times(10)
    @new_program_scope
    def test_simple(self):
        width = 10
        dat = fluid.layers.data(name='data', shape=[width], lod_level=1)
        dat.stop_gradient = False
        table = fluid.layers.lod_rank_table(dat)
        array = fluid.layers.lod_tensor_to_array(dat, table)
        restored = fluid.layers.array_to_lod_tensor(array, table)
        last = fluid.layers.sequence_pool(restored, 'last')
        loss = fluid.layers.mean(x=last)
        fluid.backward.append_backward_ops(loss)
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        rand_tensor = rand_lod_tensor(
            max_seq_len=10, max_num_seq=15, shape=[width], place=cpu)
        ig = exe.run(feed={'data': rand_tensor},
                     fetch_list=['data@GRAD'],
                     return_numpy=False)[0]
        lod = rand_tensor.lod()[0]
        ig = numpy.array(ig)
        num_seq = len(lod) - 1
        for row_id, row in enumerate(ig):
            if row_id + 1 in lod:
                self.assertTrue(numpy.alltrue(row == 1.0 / num_seq / 10))
            else:
                self.assertTrue(numpy.alltrue(row == 0.0))


if __name__ == '__main__':
    unittest.main()
