import logging
import paddle.v2.framework.core as core
import unittest
import numpy as np
from paddle.v2.framework.op import Operator, RecurrentOp
from op_test import get_numeric_gradient


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


def create_tensor(scope, name, shape, np_data):
    tensor = scope.new_var(name).get_tensor()
    tensor.set_dims(shape)
    tensor.set(np_data, core.CPUPlace())
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

    input_dim = 30
    batch_size = 50
    weight_dim = 15
    sent_len = 11

    def setUp(self):
        self.py_rnn = PySimpleRNN(self.input_dim, self.batch_size,
                                  self.weight_dim, self.sent_len)

    def forward(self):
        self.scope = core.Scope()
        self.create_global_variables()
        self.create_rnn_op()
        self.create_step_net()
        ctx = core.DeviceContext.create(core.CPUPlace())
        self.rnnop.run(self.scope, ctx)
        return np.array(self.scope.find_var("h@mem").get_tensor()).astype(
            "float32")

    def create_global_variables(self):
        # create inlink
        x_np_data = self.py_rnn.x
        create_tensor(self.scope, "x",
                      [self.sent_len, self.batch_size, self.input_dim],
                      x_np_data)
        W_np_data = self.py_rnn.W
        create_tensor(self.scope, "W", [self.input_dim, self.input_dim],
                      W_np_data)

        U_np_data = self.py_rnn.U
        create_tensor(self.scope, "U", [self.input_dim, self.input_dim],
                      U_np_data)

        h_boot_np_data = self.py_rnn.h_boot
        create_tensor(self.scope, "h_boot", [self.batch_size, self.input_dim],
                      h_boot_np_data)
        self.scope.new_var("step_scopes")
        self.scope.new_var("h@mem")

    def create_rnn_op(self):
        # create RNNOp
        self.rnnop = RecurrentOp(
            # inputs
            inlinks=["x"],
            boot_memories=["h_boot"],
            step_net="stepnet",
            # outputs
            outlinks=["h@mem"],
            step_scopes="step_scopes",
            # attributes
            pre_memories=["h@pre"],
            memories=["h@mem"])

    def create_step_net(self):
        stepnet = core.Net.create()
        x_fc_op = Operator("mul", X="x", Y="W", Out="Wx")
        h_fc_op = Operator("mul", X="h@pre", Y="U", Out="Uh")
        sum_op = Operator("sum", X=["Wx", "Uh"], Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@mem")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            stepnet.append_op(op)
        stepnet.complete_add_op(True)
        self.rnnop.set_stepnet(stepnet)

    def test_forward(self):
        print 'test recurrent op forward'
        pd_output = self.forward()
        py_output = self.py_rnn.forward()
        print 'pd_output', pd_output
        print
        print 'py_output', py_output
        self.assertEqual(pd_output.shape, py_output.shape)
        self.assertTrue(np.isclose(pd_output, py_output, rtol=0.1).all())


class RecurrentGradientOpTest(unittest.TestCase):
    def create_forward_op(self):
        self.forward_op = RecurrentOp(
            # inputs
            inlinks=["x"],
            boot_memories=["h_boot"],
            step_net="stepnet",
            # outputs
            outlinks=["h"],
            step_scopes="step_scopes",
            # attributes
            pre_memories=["h@pre"],
            memories=["h@alias"])

        # create a stepnet for RNN
        stepnet = core.Net.create()
        x_fc_op = Operator("mul", X="x@alias", Y="W", Out="Wx")
        h_fc_op = Operator("mul", X="h@pre", Y="U", Out="Uh")
        sum_op = Operator("sum", X=["Wx", "Uh"], Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@alias")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            stepnet.append_op(op)
        stepnet.complete_add_op(True)
        self.forward_op.set_stepnet(stepnet)

    def create_gradient_op(self):
        a = set()
        backward_op = core.RecurrentOp.backward(self.forward_op, a)

    def test_grad(self):
        self.create_forward_op()
        self.create_gradient_op()


if __name__ == '__main__':
    unittest.main()
