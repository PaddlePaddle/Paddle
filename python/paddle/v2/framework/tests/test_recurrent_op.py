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


class PySimpleRNN(object):
    '''
    A simple implementation of RNN based on numpy, to futhur test RecurrentOp's alogorithm
    '''

    def __init__(self, input_dim=30, batch_size=50, weight_dim=15, sent_len=11):
        self.x = np.random.normal(size=(sent_len, batch_size,
                                        input_dim)).astype("float32")
        self.W = np.random.normal(size=(input_dim, input_dim)).astype("float32")
        self.U = np.random.normal(size=(input_dim, input_dim)).astype("float32")
        self.h_boot = np.random.normal(size=(batch_size,
                                             input_dim)).astype("float32")

        # memories
        self.mems = [
            np.zeros(shape=(batch_size, input_dim)).astype("float32")
            for i in range(sent_len)
        ]

    def forward(self):
        xs = self.segment_inputs()
        for step_id in range(self.x.shape[0]):
            self.step(step_id, xs[step_id])
        return self.concat_outputs()

    def segment_inputs(self):
        return [self.x[i] for i in range(self.x.shape[0])]

    def concat_outputs(self):
        return np.array(self.mems).astype("float32")

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


class PySimpleRNNTest(unittest.TestCase):
    def setUp(self):
        self.rnn = PySimpleRNN()

    def test_forward(self):
        output = self.rnn.forward()


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

    input_dim = 10
    batch_size = 12
    weight_dim = 15
    sent_len = 11

    def setUp(self):
        self.py_rnn = PySimpleRNN(self.input_dim, self.batch_size,
                                  self.weight_dim, self.sent_len)
        self.output = self.create_rnn_op()
        #print g_program
        append_backward_ops(self.output)

    def create_rnn_op(self):
        x = data(
            shape=[self.sent_len, self.input_dim],
            data_type='float32',
            name='x')
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

        feed_map = {}
        feed_map["x"] = create_tensor(self.py_rnn.x, place)
        feed_map["W"] = create_tensor(self.py_rnn.W, place)
        feed_map["U"] = create_tensor(self.py_rnn.U, place)
        feed_map["h_boot"] = create_tensor(self.py_rnn.h_boot, place)

        exe = Executor(place)
        out = exe.run(g_program, feed=feed_map, fetch_list=[self.output])

        return np.array(out[0])

    def test_backward(self):
        tmp = self.get_numerical_gradient()

    # def test_forward(self):
    #     print 'test recurrent op forward'
    #     pd_output = self.forward()
    #     py_output = self.py_rnn.forward()
    #     print 'pd_output', pd_output
    #     print
    #     print 'py_output', py_output
    #     self.assertEqual(pd_output.shape, py_output.shape)
    #     self.assertTrue(np.isclose(pd_output, py_output, rtol=0.1).all())

    def get_numerical_gradient(self, delta=0.005):
        py_output = self.py_rnn.forward()
        dloss_dout = np.random.normal(size=py_output.shape).astype("float32")
        feed_list = [
            self.py_rnn.x, self.py_rnn.W, self.py_rnn.U, self.py_rnn.h_boot
        ]
        grad_list = [np.zeros_like(x) for x in feed_list]
        for feed, grad in zip(feed_list, grad_list):
            for f, g in np.nditer([feed, grad], op_flags=['readwrite']):
                o = f
                f[...] = o + delta
                y_pos = self.forward()
                f[...] = o - delta
                y_neg = self.forward()
                f[...] = o
                dout_dfeed = (y_pos - y_neg) / delta / 2
                g[...] = np.sum(dloss_dout * dout_dfeed)

        return grad_list


# class RecurrentGradientOpTest(unittest.TestCase):
#     def create_forward_op(self):
#         self.forward_op = RecurrentOp(
#             # inputs
#             inputs=["x"],
#             initial_states=["h_boot"],
#             step_net="stepnet",
#             # outputs
#             outputs=["h"],
#             step_scopes="step_scopes",
#             # attributes
#             ex_states=["h@pre"],
#             states=["h@alias"])
# 
#         # create a stepnet for RNN
#         stepnet = core.Net.create()
#         x_fc_op = Operator("mul", X="x@alias", Y="W", Out="Wx")
#         h_fc_op = Operator("mul", X="h@pre", Y="U", Out="Uh")
#         sum_op = Operator("sum", X=["Wx", "Uh"], Out="sum")
#         sig_op = Operator("sigmoid", X="sum", Y="h@alias")
# 
#         for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
#             stepnet.append_op(op)
#         stepnet.complete_add_op(True)
#         self.forward_op.set_stepnet(stepnet)
# 
#     def create_gradient_op(self):
#         a = set()
#         backward_op = core.RecurrentOp.backward(self.forward_op, a)
# 
#     def test_grad(self):
#         self.create_forward_op()
#         self.create_gradient_op()

if __name__ == '__main__':
    # exit(
    #     0
    # )  # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957
    unittest.main()
