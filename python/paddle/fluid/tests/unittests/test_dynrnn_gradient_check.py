#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import collections
import paddle
import paddle.fluid as fluid
import unittest
from decorator_helper import prog_scope


class Memory:
    def __init__(self, shape, dtype='float32'):
        self.ex = np.zeros(shape=shape, dtype=dtype)
        self.cur = None

    def update(self, val):
        assert val.shape == self.ex.shape
        assert val.dtype == self.ex.dtype
        self.cur = val

    def next(self):
        self.ex = self.cur
        self.cur = None

    def __next__(self):
        self.next()

    def reset(self):
        self.ex = np.zeros(shape=self.ex.shape, dtype=self.ex.dtype)
        self.cur = None


class Output:
    def __init__(self):
        self.outs = []

    def next_sequence(self):
        self.outs.append([])

    def out(self, val):
        self.outs[-1].append(val)

    def last(self):
        return self.outs[-1][-1]


class BaseRNN:
    def __init__(self, ins, mems, params, outs, num_seq=5, max_seq_len=15):
        self.num_seq = num_seq
        self.inputs = collections.defaultdict(list)

        for _ in range(num_seq):
            seq_len = random.randint(1, max_seq_len - 1)
            for iname in ins:
                ishape = ins[iname].get('shape', None)
                idtype = ins[iname].get('dtype', 'float32')
                lst = []
                for _ in range(seq_len):
                    lst.append(np.random.random(size=ishape).astype(idtype))
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
            self.params[pname] = np.random.random(size=pshape).astype(pdtype)

        self.outputs = dict()

        for oname in outs:
            self.outputs[oname] = Output()

    def step(self, **kwargs):
        raise NotImplementedError()

    def exe(self):
        retv = dict()
        for out in self.outputs:
            retv[out] = []

        for seq_id in range(self.num_seq):
            for mname in self.mems:
                self.mems[mname].reset()
            for out in self.outputs:
                self.outputs[out].next_sequence()

            iname0 = list(self.inputs.keys())[0]
            seq_len = len(self.inputs[iname0][seq_id])

            for step_id in range(seq_len):
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
            retv[out] = np.array(retv[out])
        return retv

    def to_feed(self, place):
        feed_dict = dict()

        for iname in self.inputs:
            lod = []
            np_flatten = []
            for seq_id in range(len(self.inputs[iname])):
                seq_len = len(self.inputs[iname][seq_id])
                lod.append(seq_len)
                np_flatten.extend(self.inputs[iname][seq_id])

            t = fluid.Tensor()
            t.set(np.array(np_flatten), place)
            t.set_recursive_sequence_lengths([lod])
            feed_dict[iname] = t

        for pname in self.params:
            feed_dict[pname] = self.params[pname]
        return feed_dict

    def get_numeric_gradient_of_param(self, param_name, delta=0.001):
        p = self.params[param_name]
        if len(p.shape) != 2:
            raise ValueError(
                "Not support get numeric gradient of an parameter,"
                " which is not matrix"
            )
        g = np.zeros(shape=p.shape, dtype=p.dtype)

        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                o = p[i][j]
                p[i][j] += delta
                pos = self._exe_mean_out_()
                p[i][j] -= 2 * delta
                neg = self._exe_mean_out_()
                p[i][j] = o
                g[i][j] = (pos - neg) / (delta * 2)
        return g

    def get_numeric_gradient_of_input(
        self, input_name, delta=0.001, return_one_tensor=True
    ):
        ipt = self.inputs[input_name]
        grad = []

        for seq in ipt:
            seq_grad = []
            for item in seq:
                item_grad = np.zeros(shape=item.shape, dtype=item.dtype)
                if len(item.shape) != 1:
                    raise ValueError("Not support")

                for i in range(len(item)):
                    o = item[i]
                    item[i] += delta
                    pos = self._exe_mean_out_()
                    item[i] -= 2 * delta
                    neg = self._exe_mean_out_()
                    item[i] = o
                    item_grad[i] = (pos - neg) / (delta * 2)
                seq_grad.append(item_grad)
            grad.append(seq_grad)

        if not return_one_tensor:
            return grad

        for i in range(len(grad)):
            grad[i] = np.concatenate(grad[i])
        grad = np.concatenate(grad)
        return grad

    def _exe_mean_out_(self):
        outs = self.exe()
        return np.array([o.mean() for o in outs.values()]).mean()


class SeedFixedTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Fix random seeds to remove randomness from tests"""
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        np.random.seed(123)
        random.seed(124)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)


class TestSimpleMul(SeedFixedTestCase):
    DATA_NAME = 'X'
    DATA_WIDTH = 32
    PARAM_NAME = 'W'
    HIDDEN_WIDTH = 10
    OUT_NAME = 'Out'

    class SimpleMul(BaseRNN):
        def __init__(self):
            base = TestSimpleMul
            super().__init__(
                {base.DATA_NAME: {'shape': [base.DATA_WIDTH]}},
                {},
                {
                    base.PARAM_NAME: {
                        'shape': [base.DATA_WIDTH, base.HIDDEN_WIDTH]
                    }
                },
                [base.OUT_NAME],
            )

        def step(self, X, W, Out):
            Out.out(np.matmul(X, W))

    # Test many times in local to ensure the random seed cannot breaks CI
    # @many_times(10)
    @prog_scope()
    def test_forward_backward(self):
        py_rnn = TestSimpleMul.SimpleMul()
        dat = fluid.layers.data(
            name=self.DATA_NAME, shape=[self.DATA_WIDTH], lod_level=1
        )
        dat.stop_gradient = False

        rnn = fluid.layers.DynamicRNN()
        with rnn.block():
            d = rnn.step_input(dat)
            o = fluid.layers.fc(
                input=d,
                param_attr=self.PARAM_NAME,
                bias_attr=False,
                size=self.HIDDEN_WIDTH,
                act=None,
            )
            rnn.output(o)

        out = rnn()
        out = fluid.layers.sequence_pool(out, pool_type='last')
        loss = paddle.mean(out)
        fluid.backward.append_backward(loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        out, w_g, i_g = list(
            map(
                np.array,
                exe.run(
                    feed=py_rnn.to_feed(cpu),
                    fetch_list=[
                        out,
                        self.PARAM_NAME + "@GRAD",
                        self.DATA_NAME + "@GRAD",
                    ],
                    return_numpy=False,
                ),
            )
        )
        out_by_python = py_rnn.exe()[self.OUT_NAME]
        np.testing.assert_allclose(out, out_by_python, rtol=1e-05)
        w_g_num = py_rnn.get_numeric_gradient_of_param(self.PARAM_NAME)
        np.testing.assert_allclose(w_g_num, w_g, rtol=0.05)
        i_g_num = py_rnn.get_numeric_gradient_of_input(
            input_name=self.DATA_NAME
        )
        i_g_num = i_g_num.reshape(i_g.shape)
        np.testing.assert_allclose(i_g_num, i_g, rtol=0.05)


class TestSimpleMulWithMemory(SeedFixedTestCase):
    DATA_WIDTH = 32
    HIDDEN_WIDTH = 20
    DATA_NAME = 'X'
    PARAM_NAME = 'W'

    class SimpleMulWithMemory(BaseRNN):
        def __init__(self):
            super().__init__(
                {
                    TestSimpleMulWithMemory.DATA_NAME: {
                        'shape': [TestSimpleMulWithMemory.DATA_WIDTH]
                    }
                },
                {'Mem': {'shape': [TestSimpleMulWithMemory.HIDDEN_WIDTH]}},
                {
                    TestSimpleMulWithMemory.PARAM_NAME: {
                        'shape': [
                            TestSimpleMulWithMemory.DATA_WIDTH,
                            TestSimpleMulWithMemory.HIDDEN_WIDTH,
                        ]
                    }
                },
                ['Out'],
            )

        def step(self, X, Mem, W, Out):
            o = np.matmul(X, W)
            assert isinstance(Mem, Memory)
            o += Mem.ex
            Mem.update(o)
            assert isinstance(Out, Output)
            Out.out(o)

    # many_times used locally for debug. Make sure the calculation is stable.
    # @many_times(10)
    @prog_scope()
    def test_forward_backward(self):
        py_rnn = TestSimpleMulWithMemory.SimpleMulWithMemory()
        data = fluid.layers.data(
            name=self.DATA_NAME, shape=[self.DATA_WIDTH], lod_level=1
        )
        data.stop_gradient = False
        rnn = fluid.layers.DynamicRNN()
        with rnn.block():
            d = rnn.step_input(data)
            mem = rnn.memory(value=0.0, shape=[self.HIDDEN_WIDTH])
            hidden = fluid.layers.fc(
                input=d,
                size=self.HIDDEN_WIDTH,
                param_attr=self.PARAM_NAME,
                bias_attr=False,
                act=None,
            )
            o = fluid.layers.elementwise_add(x=hidden, y=mem)
            rnn.update_memory(mem, o)
            rnn.output(o)

        out = rnn()
        last = fluid.layers.sequence_pool(input=out, pool_type='last')
        loss = paddle.mean(last)
        fluid.backward.append_backward(loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        feed = py_rnn.to_feed(cpu)
        last_np, w_g, i_g = list(
            map(
                np.array,
                exe.run(
                    feed=feed,
                    fetch_list=[
                        last,
                        self.PARAM_NAME + "@GRAD",
                        self.DATA_NAME + "@GRAD",
                    ],
                    return_numpy=False,
                ),
            )
        )
        (last_by_py,) = list(py_rnn.exe().values())
        w_g_num = py_rnn.get_numeric_gradient_of_param(self.PARAM_NAME)
        np.testing.assert_allclose(last_np, last_by_py, rtol=1e-05)

        np.testing.assert_allclose(w_g_num, w_g, rtol=0.1)
        i_g_num = py_rnn.get_numeric_gradient_of_input(self.DATA_NAME)
        i_g_num = i_g_num.reshape(i_g.shape)

        # Since this RNN has many float add. The number could be not stable.
        # rtol = 0.1
        np.testing.assert_allclose(i_g_num, i_g, rtol=0.1)


if __name__ == '__main__':
    unittest.main()
