import unittest

import logging

from op_test import get_numeric_gradient
from paddle.v2.framework.layers import *
from paddle.v2.framework.framework import g_program, g_init_program
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.backward import append_backward_ops
import numpy as np
import paddle.v2.framework.core as core


def py_sigmoid(x):
    return 1. / (1. + np.exp(-x))


class PyRNNBase(object):
    def __init__(self, input_shape, output_shape):
        self.x = np.random.normal(size=input_shape).astype("float32")
        self.y = np.zeros(shape=output_shape).astype("float32")

    def step(self):
        pass

    def forward(self):
        for step_id in range(self.x.shape[0]):
            self.step(step_id, self.x[step_id])
        return self.y

    def segment_inputs(self):
        return [self.x[i] for i in range(self.x.shape[0])]


class PySimpleRNN(PyRNNBase):
    def __init__(self, input_shape, output_shape):
        super(PySimpleRNN, self).__init__(input_shape, output_shape)

        seq_len, batch_size, input_dim = input_shape
        seq_len, batch_size, _ = output_shape

        self.W = np.random.normal(size=(input_dim, input_dim)).astype("float32")
        self.U = np.random.normal(size=(input_dim, input_dim)).astype("float32")
        self.h_boot = np.random.normal(size=(batch_size,
                                             input_dim)).astype("float32")

        men_dim = (seq_len, batch_size, input_dim)
        self.mems = np.zeros(shape=men_dim).astype("float32")

    def step(self, step_id, x):
        '''
        run a step
        '''
        mem = self.mems[step_id]
        if step_id > 0:
            pre_mem = self.mems[step_id - 1]
        else:
            pre_mem = self.h_boot
        xW = np.matmul(x, self.W).astype("float32")
        hU = np.matmul(pre_mem, self.U).astype("float32")

        sum = xW + hU
        self.mems[step_id] = py_sigmoid(sum)
        self.y[step_id] = self.mems[step_id]


def create_tensor(np_data, place):
    tensor = core.LoDTensor()
    tensor.set(np_data, place)
    return tensor


class RecurrentOpTest(unittest.TestCase):
    '''
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
    batch_size = 3
    sent_len = 2

    def setUp(self):
        self.data_field = {"x", "W", "U", "h_boot"}

        input_shape = (self.sent_len, self.batch_size, self.input_dim)
        output_shape = (self.sent_len, self.batch_size, self.input_dim)
        self.py_rnn = PySimpleRNN(input_shape, output_shape)

        self.output = self.create_rnn_op()
        loss = mean(x=self.output)
        # append_backward_ops(loss)

    def create_rnn_op(self):
        x = data(
            shape=[self.sent_len, self.batch_size, self.input_dim],
            data_type='float32',
            name='x',
            append_batch_size=False)
        h_boot = data(
            shape=[self.input_dim], data_type='float32', name='h_boot')

        rnn = StaticRNN()
        with rnn.step():
            h_pre = rnn.memory(init=h_boot)
            x_t = rnn.step_input(x)

            temp_l = fc(input=x_t,
                        size=self.input_dim,
                        param_attr={'name': 'W'},
                        bias_attr=False)
            temp_r = fc(input=h_pre,
                        size=self.input_dim,
                        param_attr={'name': 'U'},
                        bias_attr=False)

            h = sigmoid(x=elementwise_add(x=temp_l, y=temp_r))

            rnn.update_memory(h_pre, h)
            rnn.output(h)

        return rnn()

    def forward(self):
        place = core.CPUPlace()

        feed_map = {
            x: create_tensor(getattr(self.py_rnn, x), place)
            for x in self.data_field
        }

        exe = Executor(place)
        out = exe.run(g_program, feed=feed_map, fetch_list=[self.output])

        return np.array(out[0])

    # def test_backward(self):
    #     tmp = self.get_numerical_gradient()

    def test_forward(self):
        print 'test recurrent op forward'
        pd_output = self.forward()
        py_output = self.py_rnn.forward()
        print 'pd_output', pd_output
        print
        print 'py_output', py_output
        self.assertEqual(pd_output.shape, py_output.shape)
        self.assertTrue(np.isclose(pd_output, py_output, rtol=0.1).all())

    def get_numerical_gradient(self, delta=0.005):
        py_output = self.py_rnn.forward()
        dloss_dout = np.random.normal(size=py_output.shape).astype("float32")
        feed_list = [getattr(self.py_rnn, x) for x in self.data_field]
        grad_list = [np.zeros_like(x) for x in feed_list]
        for feed, grad in zip(feed_list, grad_list):
            for f, g in np.nditer([feed, grad], op_flags=['readwrite']):
                o = f
                f[...] = o + delta
                y_pos = self.py_rnn.forward()
                f[...] = o - delta
                y_neg = self.py_rnn.forward()
                f[...] = o
                dout_dfeed = (y_pos - y_neg) / delta / 2
                g[...] = np.sum(dloss_dout * dout_dfeed)

        return grad_list


if __name__ == '__main__':
    # exit(
    #     0
    # )  # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957
    unittest.main()
