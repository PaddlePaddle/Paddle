import logging
import paddle.v2.framework.core as core
import unittest
from paddle.v2.framework.op import Operator, DynamicRecurrentOp
import numpy as np

# for siplicity, just one level LoD
lod_py = [[0, 4, 7, 9, 10]]
input_dim = 30
num_sents = len(lod_py[0]) - 1
weight_dim = 15


def create_tensor(scope, name, shape, np_data):
    tensor = scope.var(name).get_tensor()
    tensor.set_dims(shape)
    tensor.set(np_data, core.CPUPlace())
    return tensor


class PyRNNStep(object):
    def __init__(self):

        self.x = np.random.normal(size=(lod_py[0][-1],
                                        input_dim)).astype("float32")
        self.W = np.random.normal(size=(input_dim, input_dim)).astype("float32")
        self.U = np.random.normal(size=(input_dim, input_dim)).astype("float32")
        self.h_boot = np.random.normal(size=(num_sents,
                                             input_dim)).astype("float32")


class DynamicRecurrentOpTest(unittest.TestCase):
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

    py = PyRNNStep()

    def forward(self):
        self.scope = core.Scope()
        self.create_global_variables()
        self.create_rnn_op()
        self.create_step_net()
        ctx = core.DeviceContext.create(core.CPUPlace())
        self.rnnop.run(self.scope, ctx)
        state = self.rnnop.get_state("h@mem")
        print 'state size: ', state.size()

        step_inputs = self.rnnop.get_step_input("x")
        print "x size ", step_inputs.size()
        for i in range(step_inputs.size()):
            print "x %d" % i, np.array(step_inputs.read(i).get_dims())
        step_outputs = self.rnnop.get_step_output('h@mem')
        print 'step_outputs.size ', step_outputs.size()
        output = self.scope.find_var("h@mem").get_tensor()
        print 'output', np.array(output).shape

    def create_global_variables(self):
        # create inlink
        x_tensor = create_tensor(self.scope, "x", [num_sents, input_dim],
                                 self.py.x)
        x_tensor.set_lod(lod_py)
        create_tensor(self.scope, "W", [input_dim, input_dim], self.py.W)
        create_tensor(self.scope, "U", [input_dim, input_dim], self.py.U)
        create_tensor(self.scope, "h_boot", [num_sents, input_dim],
                      self.py.h_boot)
        self.scope.var("step_scopes")
        self.scope.var("h@mem")

    def create_rnn_op(self):
        # create RNNOp
        self.rnnop = DynamicRecurrentOp(
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
        print 'pd_output', pd_output


class RecurrentGradientOpTest(unittest.TestCase):
    py = PyRNNStep()

    def create_forward_op(self):
        # create RNNOp
        self.forward_op = DynamicRecurrentOp(
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

    def create_gradient_op(self):
        a = set()
        backward_op = core.DynamicRecurrentOp.backward(self.forward_op, a)

    def create_step_net(self):
        stepnet = core.Net.create()
        x_fc_op = Operator("mul", X="x", Y="W", Out="Wx")
        h_fc_op = Operator("mul", X="h@pre", Y="U", Out="Uh")
        sum_op = Operator("sum", X=["Wx", "Uh"], Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@mem")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            stepnet.append_op(op)
        stepnet.complete_add_op(True)
        self.forward_op.set_stepnet(stepnet)

    def create_global_variables(self):
        # create inlink
        x_tensor = create_tensor(self.scope, "x", [num_sents, input_dim],
                                 self.py.x)
        x_tensor.set_lod(lod_py)
        create_tensor(self.scope, "W", [input_dim, input_dim], self.py.W)
        create_tensor(self.scope, "U", [input_dim, input_dim], self.py.U)
        create_tensor(self.scope, "h_boot", [num_sents, input_dim],
                      self.py.h_boot)
        self.scope.var("step_scopes")
        self.scope.var("h@mem")

    def test_grad(self):
        self.scope = core.Scope()
        self.create_forward_op()
        self.create_global_variables()
        self.create_step_net()
        self.create_gradient_op()


if __name__ == '__main__':
    unittest.main()
