import numpy
import random
import collections
import paddle.v2.fluid as fluid
import unittest
from decorators import *


class Memory(object):
    def __init__(self, shape, dtype='float32'):
        self.ex = numpy.zeros(shape=shape, dtype=dtype)
        self.cur = None

    def update(self, val):
        assert val.shape == self.ex.shape
        assert val.dtype == self.ex.dtype
        self.cur = val

    def ex(self):
        return self.ex

    def next(self):
        self.ex = self.cur
        self.cur = None

    def __next__(self):
        self.next()

    def reset(self):
        self.ex = numpy.zeros(shape=self.ex.shape, dtype=self.ex.dtype)
        self.cur = None


class Output(object):
    def __init__(self):
        self.outs = []

    def next_sequence(self):
        self.outs.append([])

    def out(self, val):
        self.outs[-1].append(val)

    def last(self):
        return self.outs[-1][-1]


class BaseRNN(object):
    def __init__(self, ins, mems, params, outs, num_seq=5, max_seq_len=15):
        self.num_seq = num_seq
        self.inputs = collections.defaultdict(list)

        for _ in xrange(num_seq):
            seq_len = random.randint(1, max_seq_len - 1)
            for iname in ins:
                ishape = ins[iname].get('shape', None)
                idtype = ins[iname].get('dtype', 'float32')
                lst = []
                for _ in xrange(seq_len):
                    lst.append(numpy.random.random(size=ishape).astype(idtype))
                self.inputs[iname].append(lst)

        self.mems = dict()
        for mname in mems:
            mshape = mems[mname].get('shape', None)
            mdtype = mems[mname].get('dtype', 'float32')
            self.mems[mname] = Memory(shape=mshape, dtype=mdtype)

        self.params = dict()
        for pname in params:
            pshape = params[pname].get('shape', None)
            pdtype = params[pname].get('dtype', 'float32')
            self.params[pname] = numpy.random.random(size=pshape).astype(pdtype)

        self.outputs = dict()

        for oname in outs:
            self.outputs[oname] = Output()

    def step(self, **kwargs):
        raise NotImplementedError()

    def exe(self):
        retv = dict()
        for out in self.outputs:
            retv[out] = []

        for seq_id in xrange(self.num_seq):
            for mname in self.mems:
                self.mems[mname].reset()
            for out in self.outputs:
                self.outputs[out].next_sequence()

            iname0 = self.inputs.keys()[0]
            seq_len = len(self.inputs[iname0][seq_id])

            for step_id in xrange(seq_len):
                xargs = dict()

                for iname in self.inputs:
                    xargs[iname] = self.inputs[iname][seq_id][step_id]

                for mname in self.mems:
                    xargs[mname] = self.mems[mname]

                for pname in self.params:
                    xargs[pname] = self.params[pname]

                for out in self.outputs:
                    xargs[out] = self.outputs[out]

                self.step(**xargs)

                for mname in self.mems:
                    next(self.mems[mname])

            for out in self.outputs:
                retv[out].append(self.outputs[out].last())

        for out in retv:
            retv[out] = numpy.array(retv[out])
        return retv

    def to_feed(self, place):
        feed_dict = dict()

        for iname in self.inputs:
            lod = [0]
            np_flatten = []
            for seq_id in xrange(len(self.inputs[iname])):
                seq_len = len(self.inputs[iname][seq_id])
                lod.append(lod[-1] + seq_len)
                np_flatten.extend(self.inputs[iname][seq_id])

            t = fluid.Tensor()
            t.set(numpy.array(np_flatten), place)
            t.set_lod([lod])
            feed_dict[iname] = t

        for pname in self.params:
            feed_dict[pname] = self.params[pname]
        return feed_dict

    def get_numeric_gradient_of_param(self, param_name, delta=0.001):
        if len(p.shape) != 2:
            raise ValueError("Not support get numeric gradient of an parameter,"
                             " which is not matrix")
        p = self.params[param_name]
        g = numpy.zeros(shape=p.shape, dtype=p.dtype)

        for i in xrange(p.shape[0]):
            for j in xrange(p.shape[1]):
                o = p[i][j]
                p[i][j] += delta
                pos = self._exe_mean_out_()
                p[i][j] -= 2 * delta
                neg = self._exe_mean_out_()
                p[i][j] = o
                g[i][j] = (pos - neg) / (delta * 2)
        return g

    def _exe_mean_out_(self):
        outs = self.exe()
        return numpy.array([o.mean() for o in outs.itervalues()]).mean()


class SimpleMul(BaseRNN):
    def __init__(self):
        super(SimpleMul, self).__init__({
            'X': {
                'shape': [32]
            }
        }, {}, {'W': {
            'shape': [32, 10]
        }}, ['Out'])

    def step(self, X, W, Out):
        Out.out(numpy.matmul(X, W))


class TestSimpleMul(unittest.TestCase):
    # Test many times in local to ensure the random seed cannot breaks CI
    # @many_times(10)
    @prog_scope()
    def test_forward_backward(self):
        python_impl = SimpleMul()
        dat = fluid.layers.data(name='X', shape=[32], lod_level=1)

        rnn = fluid.layers.DynamicRNN()
        with rnn.block():
            d = rnn.step_input(dat)
            o = fluid.layers.fc(input=d,
                                param_attr='W',
                                bias_attr=False,
                                size=10,
                                act=None)
            rnn.output(o)

        out = rnn()
        out = fluid.layers.sequence_pool(out, pool_type='last')
        loss = fluid.layers.mean(x=out)
        fluid.backward.append_backward_ops(loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        out, w_g = exe.run(feed=python_impl.to_feed(cpu),
                           fetch_list=[out, "W@GRAD"])
        out_by_python = python_impl.exe()['Out']
        self.assertTrue(numpy.allclose(out, out_by_python))
        w_g_num = python_impl.get_numeric_gradient_of_param("W")
        self.assertTrue(numpy.allclose(w_g_num, w_g, rtol=0.05))


if __name__ == '__main__':
    unittest.main()
