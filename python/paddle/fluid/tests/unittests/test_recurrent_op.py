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

import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid import ParamAttr
from paddle.fluid.backward import append_backward
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, grad_var_name

np.random.seed(123)


class PyRNNBase:
    def __init__(self, input_shape, output_shape):
        self.x = np.ones(shape=input_shape).astype("float32")
        self.y = np.zeros(shape=output_shape).astype("float32")

    def step(self, step_id, x):
        raise NotImplementedError

    def forward(self):
        for step_id in range(self.x.shape[0]):
            self.step(step_id, self.x[step_id])
        return np.array([np.mean(self.y)])

    def segment_inputs(self):
        return [self.x[i] for i in range(self.x.shape[0])]


class PySimpleRNN1(PyRNNBase):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)

        seq_len, batch_size, input_dim = input_shape
        self.h_boot = np.random.normal(size=(batch_size, input_dim)).astype(
            "float32"
        )

        self.scale = 1.0 / 2.0
        men_dim = (seq_len, batch_size, input_dim)
        self.mems = np.zeros(shape=men_dim).astype("float32")

    def step(self, step_id, x):
        if step_id == 0:
            pre_mem = self.h_boot
        else:
            pre_mem = self.mems[step_id - 1]
        self.mems[step_id] = (pre_mem + x) * self.scale
        self.y[step_id] = self.mems[step_id]


class PySimpleRNN2(PyRNNBase):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)

        seq_len, batch_size, input_dim = input_shape
        self.W = np.ones(shape=(input_dim, input_dim)).astype("float32")
        self.U = np.zeros(shape=(input_dim, input_dim)).astype("float32")
        self.h_boot = np.ones(shape=(batch_size, input_dim)).astype("float32")

        men_dim = (seq_len, batch_size, input_dim)
        self.mems = np.zeros(shape=men_dim).astype("float32")

    def step(self, step_id, x):
        if step_id > 0:
            pre_mem = self.mems[step_id - 1]
        else:
            pre_mem = self.h_boot
        xW = np.matmul(x, self.W).astype("float32")
        hU = np.matmul(pre_mem, self.U).astype("float32")

        def py_sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        self.mems[step_id] = py_sigmoid(xW + hU)
        self.y[step_id] = self.mems[step_id]


def create_tensor(np_data, place):
    tensor = core.LoDTensor()
    tensor.set(np_data, place)
    return tensor


class RecurrentOpTest1(unittest.TestCase):
    '''
    Test RNNOp
    equation:
        h_t = ( x_t + h_{t-1} ) / scale
    vars:
        - x
    memories:
        - h
    outputs:
        - h
    '''

    input_dim = 2
    batch_size = 1
    sent_len = 1

    def setup_program(self):
        self.main_program = Program()
        self.startup_program = Program()
        self.place = core.CPUPlace()

    def setUp(self):
        self.setup_program()
        self.feed_data_field = {"x", "h_boot"}
        self.grad_data_field = self.feed_data_field

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = PySimpleRNN1(self.input_shape, self.output_shape)

        with fluid.program_guard(self.main_program, self.startup_program):
            self.output = paddle.mean(self.create_rnn_op())

    def create_rnn_op(self):
        x = layers.data(
            shape=[self.sent_len, self.batch_size, self.input_dim],
            dtype='float32',
            name='x',
            append_batch_size=False,
        )
        x.stop_gradient = False
        h_boot = layers.data(
            shape=[self.input_dim], dtype='float32', name='h_boot'
        )
        h_boot.stop_gradient = False

        rnn = layers.StaticRNN()
        with rnn.step():
            h_pre = rnn.memory(init=h_boot)
            x_t = rnn.step_input(x)

            h = paddle.scale(
                x=paddle.add(x=h_pre, y=x_t),
                scale=self.py_rnn.scale,
            )

            rnn.update_memory(h_pre, h)
            rnn.output(h)

        return rnn()

    def forward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), self.place)
            for x in self.feed_data_field
        }
        exe = Executor(self.place)
        out = exe.run(
            self.main_program, feed=self.feed_map, fetch_list=[self.output]
        )

        return out[0]

    def backward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), self.place)
            for x in self.feed_data_field
        }
        fetch_list = [
            self.main_program.global_block().var(grad_var_name(x))
            for x in self.grad_data_field
        ]

        exe = Executor(self.place)
        return exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=fetch_list,
            return_numpy=False,
        )

    def test_backward(self, rtol=0.01):
        self.check_forward()

        with fluid.program_guard(self.main_program, self.startup_program):
            append_backward(self.output)

        ana_grad = [np.array(x) for x in self.backward()]

        num_grad = self.get_numerical_gradient()
        for idx, name in enumerate(self.grad_data_field):
            self.assertEqual(num_grad[idx].shape, ana_grad[idx].shape)
            np.testing.assert_allclose(
                num_grad[idx],
                ana_grad[idx],
                rtol=rtol,
                atol=1e-8,
                err_msg='num_grad ('
                + name
                + ') has diff at '
                + str(self.place)
                + '\nExpect '
                + str(num_grad[idx])
                + '\n'
                + 'But Got'
                + str(ana_grad[idx])
                + ' in class '
                + self.__class__.__name__,
            )

    def check_forward(self):
        pd_output = self.forward()
        py_output = self.py_rnn.forward()
        self.assertEqual(pd_output.shape, py_output.shape)
        np.testing.assert_allclose(pd_output, py_output, rtol=0.01)

    def get_numerical_gradient(self, delta=0.005):
        dloss_dout = 1.0
        feed_list = [getattr(self.py_rnn, x) for x in self.grad_data_field]
        grad_list = [np.zeros_like(x) for x in feed_list]
        for feed, grad in zip(feed_list, grad_list):
            for f, g in np.nditer([feed, grad], op_flags=['readwrite']):
                o = float(f)
                f[...] = o + delta
                y_pos = self.forward()

                f[...] = o - delta
                y_neg = self.forward()

                f[...] = o
                dout_dfeed = (y_pos - y_neg) / (delta * 2)
                g[...] = dout_dfeed[0]

        return grad_list


class RecurrentOpTest2(RecurrentOpTest1):
    r'''
    Test RNNOp
    equation:
        h_t = \sigma (W x_t + U h_{t-1})
    weights:
        - W
        - U
    vars:
        - x
    memories:
        - h
    outputs:
       - h
    '''

    input_dim = 2
    batch_size = 10
    sent_len = 2

    def setUp(self):
        self.setup_program()

        self.feed_data_field = {"x", "h_boot", "W", "U"}
        self.grad_data_field = self.feed_data_field

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = PySimpleRNN2(self.input_shape, self.output_shape)

        with fluid.program_guard(self.main_program, self.startup_program):
            self.output = paddle.mean(self.create_rnn_op())

    def create_rnn_op(self):
        x = layers.data(
            shape=[self.sent_len, self.batch_size, self.input_dim],
            dtype='float32',
            name='x',
            append_batch_size=False,
        )
        x.stop_gradient = False
        h_boot = layers.data(
            shape=[self.input_dim], dtype='float32', name='h_boot'
        )
        h_boot.stop_gradient = False

        rnn = layers.StaticRNN()
        with rnn.step():
            h_pre = rnn.memory(init=h_boot)
            x_t = rnn.step_input(x)

            temp_l = layers.fc(
                input=x_t,
                size=self.input_dim,
                param_attr=ParamAttr(
                    name='W',
                    initializer=fluid.initializer.ConstantInitializer(1.0),
                ),
                bias_attr=False,
            )
            temp_r = layers.fc(
                input=h_pre,
                size=self.input_dim,
                param_attr=ParamAttr(
                    name='U',
                    initializer=fluid.initializer.ConstantInitializer(0.0),
                ),
                bias_attr=False,
            )

            h = paddle.nn.functional.sigmoid(x=paddle.add(x=temp_l, y=temp_r))

            rnn.update_memory(h_pre, h)
            rnn.output(h)

        return rnn()

    def test_backward(self):
        super().test_backward(rtol=0.01)


class RecurrentOpMultipleMemoryTest(RecurrentOpTest1):
    '''
    Test RNNOp with two memories
    equation:
        h_1 = h_pre_1
        h_2 = h_pre_2
        y = h_1 + h_2
    vars:
        - x
    memories:
        - h_1, h_2
    outputs:
       - y
    '''

    class PySimpleRNN3(PyRNNBase):
        def __init__(self, input_shape, output_shape):
            super().__init__(input_shape, output_shape)

            seq_len, batch_size, input_dim = input_shape
            self.h_boot1 = np.random.normal(
                size=(batch_size, input_dim)
            ).astype("float32")
            self.h_boot2 = np.random.normal(
                size=(batch_size, input_dim)
            ).astype("float32")

            men_dim = (seq_len, batch_size, input_dim)
            self.mems1 = np.zeros(shape=men_dim).astype("float32")
            self.mems2 = np.zeros(shape=men_dim).astype("float32")

        def step(self, step_id, x):
            if step_id == 0:
                pre_mem1 = self.h_boot1
                pre_mem2 = self.h_boot2
            else:
                pre_mem1 = self.mems1[step_id - 1]
                pre_mem2 = self.mems2[step_id - 1]
            self.mems1[step_id] = pre_mem1
            self.mems2[step_id] = pre_mem2
            self.y[step_id] = self.mems1[step_id] + self.mems2[step_id] + x

    input_dim = 1
    batch_size = 1
    sent_len = 2

    def setUp(self):
        self.setup_program()

        self.feed_data_field = {"x", "h_boot1", "h_boot2"}
        self.grad_data_field = self.feed_data_field

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = RecurrentOpMultipleMemoryTest.PySimpleRNN3(
            self.input_shape, self.output_shape
        )

        with fluid.program_guard(self.main_program, self.startup_program):
            self.output = paddle.mean(self.create_rnn_op())

    def create_rnn_op(self):
        x = layers.data(
            shape=[self.sent_len, self.batch_size, self.input_dim],
            dtype='float32',
            name='x',
            append_batch_size=False,
        )
        x.stop_gradient = False
        h_boot1 = layers.data(
            shape=[self.batch_size, self.input_dim],
            dtype='float32',
            name='h_boot1',
            append_batch_size=False,
        )
        h_boot1.stop_gradient = False
        h_boot2 = layers.data(
            shape=[self.batch_size, self.input_dim],
            dtype='float32',
            name='h_boot2',
            append_batch_size=False,
        )
        h_boot2.stop_gradient = False

        rnn = layers.StaticRNN()
        with rnn.step():
            h_pre1 = rnn.memory(init=h_boot1)
            h_pre2 = rnn.memory(init=h_boot2)
            x_t = rnn.step_input(x)

            mem1 = paddle.scale(x=h_pre1, scale=1.0)
            mem2 = paddle.scale(x=h_pre2, scale=1.0)
            out = layers.sums(input=[mem1, x_t, mem2])

            rnn.update_memory(h_pre1, mem1)
            rnn.update_memory(h_pre2, mem2)
            rnn.output(out)

        return rnn()


class RecurrentOpNoMemBootTest(RecurrentOpTest1):
    '''
    Test RNNOp with two memories
    equation:
        mem = x + mem_pre
        y = mem
    vars:
        - x
    memories:
        - mem
    outputs:
       - y
    '''

    class PySimpleRNN4(PyRNNBase):
        def __init__(self, input_shape, output_shape):
            super().__init__(input_shape, output_shape)
            men_dim = input_shape
            self.mems = np.zeros(shape=men_dim).astype("float32")

        def step(self, step_id, x):
            if step_id == 0:
                pre_mem = np.zeros_like(x)
            else:
                pre_mem = self.mems[step_id - 1]
            self.mems[step_id] = pre_mem + x
            self.y[step_id] = self.mems[step_id]

    input_dim = 1
    batch_size = 1
    sent_len = 2

    def setUp(self):
        self.setup_program()

        self.feed_data_field = {"x"}
        self.grad_data_field = self.feed_data_field

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = RecurrentOpNoMemBootTest.PySimpleRNN4(
            self.input_shape, self.output_shape
        )

        with fluid.program_guard(self.main_program, self.startup_program):
            self.output = paddle.mean(self.create_rnn_op())

    def create_rnn_op(self):
        x = layers.data(
            shape=[self.sent_len, self.batch_size, self.input_dim],
            dtype='float32',
            name='x',
            append_batch_size=False,
        )
        x.stop_gradient = False

        rnn = layers.StaticRNN()
        with rnn.step():
            mem_pre = rnn.memory(shape=[-1, self.input_dim], batch_ref=x)
            x_t = rnn.step_input(x)
            mem = paddle.add(x=mem_pre, y=x_t)
            rnn.update_memory(mem_pre, mem)
            rnn.output(mem)

        return rnn()


class RecurrentOpSubBlockTest(RecurrentOpTest1):
    r'''
    Test RNNOp with subblock variable
    equation:
        y_ = emb * w1
        h_t = \concat([x, h_{t-1}])
        h_t = h_t * w2
        h_t = \\unsqueeze(h_t, 1)
        h_t = \dot_attention(h_t, y_)
        h_t = \squeeze(h_t, 1)
        y = h_t
    vars:
        - x
        - w1
        - w2
    memories:
        - h
    outputs:
       - y
    '''

    class PySimpleRNN5(PyRNNBase):
        def __init__(self, input_shape, output_shape):
            super().__init__(input_shape, output_shape)

            seq_len, batch_size, input_dim = input_shape
            self.w1 = np.random.uniform(
                -0.1, 0.1, size=(input_dim, input_dim)
            ).astype("float32")
            self.w2 = np.random.uniform(
                -0.1, 0.1, size=(input_dim * 2, input_dim)
            ).astype("float32")

            self.emb = np.random.uniform(
                -0.1, 0.1, size=(seq_len, batch_size, input_dim)
            ).astype("float32")

            men_dim = (seq_len, batch_size, input_dim)
            self.mems = np.zeros(shape=men_dim).astype("float32")
            self.oy = np.matmul(self.emb, self.w1)

        def step(self, step_id, x):
            def dot_attention(query, memory):
                attn = np.matmul(query, memory.transpose((0, 2, 1)))
                weight = softmax(attn)
                weight_memory = np.matmul(weight, memory)
                return weight_memory, weight

            def softmax(x):
                return np.exp(x) / sum(np.exp(x))

            if step_id == 0:
                pre_mem = np.zeros_like(x)
            else:
                pre_mem = self.mems[step_id - 1]
            concat_in = np.concatenate([x, pre_mem], 1)
            new_mem = np.matmul(concat_in, self.w2)

            new_mem = np.expand_dims(new_mem, 1)
            new_mem, _ = dot_attention(new_mem, self.oy)
            new_mem = np.squeeze(new_mem, 1)

            self.mems[step_id] = new_mem
            self.y[step_id] = self.mems[step_id]

    input_dim = 2
    batch_size = 3
    sent_len = 3

    def setUp(self):
        self.setup_program()

        self.feed_data_field = {"x", "emb", "w1", "w2"}
        self.grad_data_field = self.feed_data_field

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = RecurrentOpSubBlockTest.PySimpleRNN5(
            self.input_shape, self.output_shape
        )

        with fluid.program_guard(self.main_program, self.startup_program):
            rnn_out = self.create_rnn_op()
            self.output = paddle.mean(rnn_out)

    def create_rnn_op(self):
        x = layers.data(
            shape=[self.sent_len, self.batch_size, self.input_dim],
            dtype='float32',
            name='x',
            append_batch_size=False,
        )
        x.stop_gradient = False

        emb = layers.data(
            name='emb',
            shape=[self.sent_len, self.batch_size, self.input_dim],
            dtype='float32',
            append_batch_size=False,
        )
        emb.stop_gradient = False

        w1 = layers.data(
            shape=[self.input_dim, self.input_dim],
            dtype='float32',
            name='w1',
            append_batch_size=False,
        )
        w1.stop_gradient = False
        w2 = layers.data(
            shape=[self.input_dim * 2, self.input_dim],
            dtype='float32',
            name='w2',
            append_batch_size=False,
        )
        w2.stop_gradient = False

        rnn = layers.StaticRNN()

        def dot_attention(query, memory):
            attn = paddle.matmul(query, memory, transpose_y=True)
            weight = paddle.nn.functional.softmax(attn)
            weight_memory = paddle.matmul(weight, memory)

            return weight_memory, weight

        y = paddle.matmul(emb, w1)
        with rnn.step():
            pre_h = rnn.memory(
                shape=(self.sent_len, self.input_dim),
                batch_ref=x,
                init_value=0.0,
            )
            step_in = rnn.step_input(x)
            concat_in = layers.concat([step_in, pre_h], 1)
            new_h = paddle.matmul(concat_in, w2)
            new_h = paddle.unsqueeze(new_h, [1])
            new_h, _ = dot_attention(new_h, y)
            new_h = paddle.squeeze(new_h, [1])

            rnn.update_memory(pre_h, new_h)
            rnn.step_output(new_h)

        return rnn()


class RecurrentOpStopGradientTest(RecurrentOpTest1):
    r"""
    Test RNNOp with stop_gradient = True
    equation:
        h_t = \sigma (W x_t + U h_{t-1})
    weights:
        - W
        - U
    vars:
        - x
    memories:
        - h
    output:
        - h
    """

    input_dim = 2
    batch_size = 10
    sent_len = 2

    def setUp(self):
        self.setup_program()
        self.feed_data_field = {"x", "h_boot", "W", "U"}
        self.grad_data_field = {"x", "W", "U"}

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = PySimpleRNN2(self.input_shape, self.output_shape)

        with fluid.program_guard(self.main_program, self.startup_program):
            self.output = paddle.mean(self.create_rnn_op())

    def create_rnn_op(self):
        x = layers.data(
            shape=[self.sent_len, self.batch_size, self.input_dim],
            dtype="float32",
            name="x",
            append_batch_size=False,
        )
        x.stop_gradient = False
        h_boot = layers.data(
            shape=[self.input_dim], dtype="float32", name="h_boot"
        )
        h_boot.stop_gradient = True

        rnn = layers.StaticRNN()
        with rnn.step():
            h_pre = rnn.memory(init=h_boot)  # init doesn't have gradient
            x_t = rnn.step_input(x)

            temp_l = layers.fc(
                input=x_t,
                size=self.input_dim,
                param_attr=ParamAttr(
                    name="W",
                    initializer=fluid.initializer.ConstantInitializer(1.0),
                ),
                bias_attr=False,
            )
            temp_r = layers.fc(
                input=h_pre,
                size=self.input_dim,
                param_attr=ParamAttr(
                    name="U",
                    initializer=fluid.initializer.ConstantInitializer(0.0),
                ),
                bias_attr=False,
            )

            h = paddle.nn.functional.sigmoid(x=paddle.add(temp_l, temp_r))

            rnn.update_memory(h_pre, h)
            rnn.output(h)

        return rnn()


if __name__ == '__main__':
    unittest.main()
