import logging
import paddle.v2.framework.core as core
import unittest
import numpy as np
import paddle.v2.framework.create_op_creation_methods as creation

ops = creation.op_creations


def create_tensor(scope, name, shape):
    tensor = scope.new_var(name).get_tensor()
    tensor.set_dims(shape)
    tensor.set(np.random.random(shape), core.CPUPlace())
    return tensor


class TestRNN(unittest.TestCase):
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

    def init(self):

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
        rnnop = ops.recurrent_op(
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

        x_fc_op = ops.fc(X="x@alias", W="W", Y="Wx")
        h_fc_op = ops.fc(X="h@pre", W="U", Y="Uh")
        sum_op = ops.add_two(X="Wx", Y="Uh", Out="sum")
        sig_op = ops.sigmoid(X="sum", Y="h@alias")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            stepnet.add_op(op)
        stepnet.complete_add_op(True)

    def test_recurrent(self):
        self.init()


if __name__ == '__main__':
    unittest.main()
