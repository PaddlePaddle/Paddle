import logging
import paddle.v2.framework.core as core
import unittest
import numpy as np
from paddle.v2.framework.op import Operator


def py_sigmoid(x):
    return 1. / (1 + np.exp(-x))


class PySimpleRNN(object):
    '''
    A simple implementation of RNN based on numpy, to futhur test RecurrentOp's alogorithm
    '''
    def __init__(self,
                 input_dim = 30,
                 batch_size = 50,
                 weight_dim = 15,
                 sent_len = 11):
        self.x = np.random.normal(size=(sent_len, batch_size, input_dim))
        self.W = np.random.normal(size=(input_dim, input_dim))
        self.U = np.random.normal(size=(input_dim, input_dim))
        self.h_boot = np.random.normal(size=(batch_size, input_dim))

        # memories
        self.mems = [np.zeros(shape=(batch_size, input_dim)) for i in range(sent_len)]

    def forward(self):
        xs = self.segment_inputs()
        for step_id in range(self.x.shape[0]):
            self.step(step_id, xs[step_id])
        return self.concat_outputs()

    def segment_inputs(self):
        return [self.x[i] for i in range(self.x.shape[0])]

    def concat_outputs(self):
        return np.array(self.mems)

    def step(self, step_id, x):
        '''
        run a step
        '''
        mem = self.mems[step_id]
        if step_id > 0:
            pre_mem = self.mems[step_id-1]
        else:
            pre_mem = self.h_boot
        xW = np.matmul(x, self.W)
        hU = np.matmul(mem, self.U)

        sum = xW + hU
        self.mems[step_id] = py_sigmoid(sum)

class PySimpleRNNTest(unittest.TestCase):
    def setUp(self):
        self.rnn = PySimpleRNN()

    def test_forward(self):
        output = self.rnn.forward()
        print 'output', output


def create_tensor(scope, name, shape):
    tensor = scope.new_var(name).get_tensor()
    tensor.set_dims(shape)
    tensor.set(np.random.random(shape), core.CPUPlace())
    return tensor


class TestRecurrentOp(unittest.TestCase):
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

    def forward(self):

        self.scope = core.Scope()

        self.create_global_variables()
        self.create_step_net()
        rnn_op = self.create_rnn_op()
        ctx = core.DeviceContext.create(core.CPUPlace())
        print 'infer_shape'
        rnn_op.infer_shape(self.scope)
        rnn_op.run(self.scope, ctx)

    def create_global_variables(self):
        # create inlink
        create_tensor(self.scope, "x",
                      [self.sent_len, self.batch_size, self.input_dim])
        create_tensor(self.scope, "W", [self.input_dim, self.input_dim])
        create_tensor(self.scope, "U", [self.input_dim, self.input_dim])
        create_tensor(self.scope, "h_boot", [self.batch_size, self.input_dim])
        self.scope.new_var("step_scopes")
        self.scope.new_var("h@alias")
        self.scope.new_var("h")

    def create_rnn_op(self):
        # create RNNOp
        rnnop = Operator("recurrent_op",
            # inputs
            inlinks=["x"],
            boot_memories=["h_boot"],
            step_net="stepnet",
            # outputs
            outlinks=["h"],
            step_scopes="step_scopes",
            # attributes
            inlink_alias=["x@alias"],
            outlink_alias=["h@alias"],
            pre_memories=["h@pre"],
            memories=["h@alias"])
        return rnnop

    def create_step_net(self):
        var = self.scope.new_var("stepnet")
        stepnet = var.get_net()

        x_fc_op = Operator("fc", X="x@alias", W="W", Y="Wx")
        h_fc_op = Operator("fc", X="h@pre", W="U", Y="Uh")
        sum_op = Operator("add_two", X="Wx", Y="Uh", Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@alias")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            stepnet.add_op(op)
        stepnet.complete_add_op(True)

    def test_forward(self):
        print 'test recurrent op forward'
        self.forward()


if __name__ == '__main__':
    unittest.main()
