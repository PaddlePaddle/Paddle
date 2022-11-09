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

import os
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import unittest

from paddle.fluid import ParamAttr
from paddle.fluid.framework import Program, grad_var_name
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward
import paddle

paddle.enable_static()

np.random.seed(123)
os.environ["CPU_NUM"] = "1"
fluid.core._set_eager_deletion_mode(0.0, 1.0, True)


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


class EagerDeletionRecurrentOpTest1(unittest.TestCase):
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
        self.data_field = {"x", "h_boot"}

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

            h = layers.scale(
                x=layers.elementwise_add(x=h_pre, y=x_t),
                scale=self.py_rnn.scale,
            )

            rnn.update_memory(h_pre, h)
            rnn.output(h)

        return rnn()

    def forward(self):
        gc_vars = core._get_eager_deletion_vars(
            self.main_program.desc, [self.output.name]
        )
        self.assertEqual(len(gc_vars), self.main_program.num_blocks)
        self.feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), self.place)
            for x in self.data_field
        }
        exe = Executor(self.place)
        out = exe.run(
            self.main_program, feed=self.feed_map, fetch_list=[self.output]
        )

        return out[0]

    def backward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), self.place)
            for x in self.data_field
        }
        fetch_list = [
            self.main_program.global_block().var(grad_var_name(x))
            for x in self.data_field
        ]

        gc_vars = core._get_eager_deletion_vars(
            self.main_program.desc, [var.name for var in fetch_list]
        )
        self.assertEqual(len(gc_vars), self.main_program.num_blocks)

        exe = Executor(self.place)
        return exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=fetch_list,
            return_numpy=False,
        )

    def test_backward(self, rtol=0.01):
        self.check_forward()
        num_grad = self.get_numerical_gradient()

        with fluid.program_guard(self.main_program, self.startup_program):
            append_backward(self.output)

        ana_grad = [np.array(x) for x in self.backward()]

        for idx, name in enumerate(self.data_field):
            self.assertEqual(num_grad[idx].shape, ana_grad[idx].shape)
            np.testing.assert_allclose(
                num_grad[idx],
                ana_grad[idx],
                rtol=rtol,
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
        feed_list = [getattr(self.py_rnn, x) for x in self.data_field]
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


class EagerDeletionRecurrentOpTest2(EagerDeletionRecurrentOpTest1):
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

        self.data_field = {"x", "h_boot", "W", "U"}

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

            h = layers.sigmoid(x=layers.elementwise_add(x=temp_l, y=temp_r))

            rnn.update_memory(h_pre, h)
            rnn.output(h)

        return rnn()

    def test_backward(self):
        super().test_backward(rtol=0.01)


class EagerDeletionRecurrentOpMultipleMemoryTest(EagerDeletionRecurrentOpTest1):
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
            super(
                EagerDeletionRecurrentOpMultipleMemoryTest.PySimpleRNN3, self
            ).__init__(input_shape, output_shape)

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

        self.data_field = {"x", "h_boot1", "h_boot2"}

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = EagerDeletionRecurrentOpMultipleMemoryTest.PySimpleRNN3(
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

            mem1 = layers.scale(x=h_pre1, scale=1.0)
            mem2 = layers.scale(x=h_pre2, scale=1.0)
            out = layers.sums(input=[mem1, x_t, mem2])

            rnn.update_memory(h_pre1, mem1)
            rnn.update_memory(h_pre2, mem2)
            rnn.output(out)

        return rnn()


class EagerDeletionRecurrentOpNoMemBootTest(EagerDeletionRecurrentOpTest1):
    '''
    Test RNNOp without memory boot
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
            super(
                EagerDeletionRecurrentOpNoMemBootTest.PySimpleRNN4, self
            ).__init__(input_shape, output_shape)
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

        self.data_field = {"x"}

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = EagerDeletionRecurrentOpNoMemBootTest.PySimpleRNN4(
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
            mem = layers.elementwise_add(x=mem_pre, y=x_t)
            rnn.update_memory(mem_pre, mem)
            rnn.output(mem)

        return rnn()


class EagerDeletionTwoRecurrentOpsTest(EagerDeletionRecurrentOpTest1):
    '''
    Test RNNOp with two recurrent ops
    equation:
        first_rnn:
            mem_inside = x + mem_pre_inside
            first_inside_out = mem_inside
        second_rnn:
            mem = x + reduce_sum(rnn_inside_out)
            y = mem + mem_pre
    vars:
        - x
    memories:
        - mem_inside
        - mem
    outputs:
       - y
    '''

    class PySimpleRNN5(PyRNNBase):
        def __init__(self, input_shape, output_shape):
            super().__init__(input_shape, output_shape)
            self.mem_0 = np.zeros(shape=input_shape).astype("float32")
            self.mem_1 = np.zeros(shape=input_shape).astype("float32")
            self.rnn_0_output = np.zeros(shape=input_shape).astype("float32")

        def step(self, step_id, x):
            # First Rnn
            for step in range(self.x.shape[0]):
                x_t = self.x[step]
                pre_mem = (
                    np.zeros_like(x_t) if step == 0 else self.mem_0[step - 1]
                )
                self.mem_0[step] = x_t + pre_mem
                self.rnn_0_output[step] = self.mem_0[step]
            # Second RNN
            pre_mem = (
                np.zeros_like(x) if step_id == 0 else self.mem_1[step_id - 1]
            )
            self.mem_1[step_id] = x + np.sum(self.rnn_0_output)
            self.y[step_id] = self.mem_1[step_id] + pre_mem

    input_dim = 1
    batch_size = 1
    sent_len = 1

    def setUp(self):
        self.setup_program()

        self.data_field = {"x"}

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = EagerDeletionTwoRecurrentOpsTest.PySimpleRNN5(
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

        rnn_0 = layers.StaticRNN()
        with rnn_0.step():
            x_t = rnn_0.step_input(x)
            mem_pre = rnn_0.memory(shape=[-1, self.input_dim], batch_ref=x)
            mem = layers.elementwise_add(x=mem_pre, y=x_t)
            rnn_0.update_memory(mem_pre, mem)
            rnn_0.output(mem)

        rnn_1 = layers.StaticRNN()
        with rnn_1.step():
            mem_pre = rnn_1.memory(shape=[-1, self.input_dim], batch_ref=x)
            x_t = rnn_1.step_input(x)
            last_rnn_output = rnn_0()
            last_rnn_sum = fluid.layers.reduce_sum(last_rnn_output)
            mem = layers.elementwise_add(x=x_t, y=last_rnn_sum)
            y = layers.elementwise_add(x=mem_pre, y=mem)
            rnn_1.update_memory(mem_pre, mem)
            rnn_1.output(y)
        return rnn_1()


class EagerDeletionRecurrentOpParallelExecutorTest(
    EagerDeletionRecurrentOpTest1
):
    '''
    Test RNNOp with ParallelExecutor
    equation:
        h_t = ( x_t + h_{t-1} ) / scale
    vars:
        - x
    memories:
        - h
    outputs:
        - h
    '''

    def forward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), self.place)
            for x in self.data_field
        }

        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        exec_strategy = fluid.ExecutionStrategy()
        parallel_exe = fluid.ParallelExecutor(
            use_cuda=False,
            main_program=self.main_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
        )
        out = parallel_exe.run(feed=self.feed_map, fetch_list=[self.output])
        return out[0]

    def backward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), self.place)
            for x in self.data_field
        }
        fetch_list = [
            self.main_program.global_block().var(grad_var_name(x))
            for x in self.data_field
        ]

        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        exec_strategy = fluid.ExecutionStrategy()
        parallel_exe = fluid.ParallelExecutor(
            use_cuda=False,
            loss_name=self.output.name,
            main_program=self.main_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
        )
        return parallel_exe.run(
            feed=self.feed_map, fetch_list=fetch_list, return_numpy=False
        )


class EagerDeletionFarwardOnlyRnnAndBackwardRnnTest(
    EagerDeletionRecurrentOpTest1
):
    '''
    Test one forward only RNN and one backward RNN in one program
    '''

    def setUp(self):
        self.setup_program()
        self.data_field = {"x", "h_boot"}

        self.input_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = PySimpleRNN1(self.input_shape, self.output_shape)

        with fluid.program_guard(self.main_program, self.startup_program):
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

            forward_only_rnn = layers.StaticRNN()
            with forward_only_rnn.step():
                h_pre = forward_only_rnn.memory(init=h_boot)
                x_t = forward_only_rnn.step_input(x)

                h = layers.scale(
                    x=layers.elementwise_add(x=h_pre, y=x_t),
                    scale=self.py_rnn.scale,
                )

                forward_only_rnn.update_memory(h_pre, h)
                forward_only_rnn.output(h)
            forward_only_output = forward_only_rnn()
            forward_only_output.stop_gradient = True
            self.forward_only_output = paddle.mean(forward_only_output)

            rnn = layers.StaticRNN()
            with rnn.step():
                h_pre = rnn.memory(init=h_boot)
                x_t = rnn.step_input(x)

                h = layers.scale(
                    x=layers.elementwise_add(x=h_pre, y=x_t),
                    scale=self.py_rnn.scale,
                )

                rnn.update_memory(h_pre, h)
                rnn.output(h)

            self.output = paddle.mean(rnn())

    def forward_two_rnn(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), self.place)
            for x in self.data_field
        }
        exe = Executor(self.place)
        out = exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=[self.forward_only_output, self.output],
        )

        return out[0], out[1]

    def check_forward(self):
        forward_only_output, pd_output = self.forward_two_rnn()
        py_output = self.py_rnn.forward()
        self.assertEqual(forward_only_output.shape, py_output.shape)
        self.assertEqual(pd_output.shape, py_output.shape)
        np.testing.assert_allclose(forward_only_output, py_output, rtol=0.01)
        np.testing.assert_allclose(pd_output, py_output, rtol=0.01)


class RecurrentNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.cell = paddle.nn.SimpleRNNCell(16, 32)
        self.rnn = paddle.nn.RNN(self.cell)

    def forward(self, inputs, prev_h):
        outputs, final_states = self.rnn(inputs, prev_h)
        return outputs, final_states


class TestDy2StRecurrentOpBackward(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.seed(100)

    def tearDown(self):
        paddle.enable_static()

    def test_recurrent_backward(self):
        net = RecurrentNet()
        inputs = paddle.rand((4, 23, 16))
        inputs.stop_gradient = False
        prev_h = paddle.randn((4, 32))
        prev_h.stop_gradient = False

        outputs, final_states = net(inputs, prev_h)
        outputs.backward()
        dy_grad = inputs.gradient()
        inputs.clear_gradient()

        net = paddle.jit.to_static(net)
        outputs, final_states = net(inputs, prev_h)
        outputs.backward()
        st_grad = inputs.gradient()
        np.testing.assert_allclose(dy_grad, st_grad)


if __name__ == '__main__':
    unittest.main()
