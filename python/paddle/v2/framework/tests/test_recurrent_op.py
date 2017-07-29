import paddle.v2.framework.core as core
import unittest
import numpy as np
import paddle.v2.framework.create_op_creation_methods as creation

ops = creation.op_creations


def create_tensor(scope, name, shape):
    tensor = scope.create_var(name).get_tensor()
    tensor.set_dims(shape)
    tensor.alloc_float()
    tensor.set(np.random.random(shape))
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

    def init(self):
        input_dim = 30
        batch_size = 50
        weight_dim = 15

        self.scope = core.Scope(None)

        # create vars
        create_tensor(self.scope, "x", [batch_size, input_dim])
        create_tensor(self.scope, "W", [input_dim, weight_dim])
        create_tensor(self.scope, "U", [weight_dim, weight_dim])
        create_tensor(self.scope, "h_boot", [batch_size, weight_dim])

        x_alias = "x@alias"
        y_alias = "y@alias"
        memory = "h@alias"
        prememory = "h@pre"
        output = "rnn_out"
        output_alias = "rnn_out@alias"

        # create step net
        stepnet_var = self.scope.create_var("stepnet")
        stepnet = stepnet_var.get_net()
        # stepnet = core.Net.create()
        x_fc_op = ops.fc(X=x_alias, W="W", Y="Wx")
        h_fc_op = ops.fc(X=prememory, W="U", Y="Uh")
        sum_op = ops.add_two(X="Wx", Y="Uh", Out="sum")
        sig_op = ops.sigmoid(X="sum", Y=memory)
        stepnet.add_op(x_fc_op)
        stepnet.add_op(h_fc_op)
        stepnet.add_op(sum_op)
        stepnet.add_op(sig_op)
        stepnet.complete_add_op(True)

        # create RNNOp
        rnnop = ops.recurrent_op(
            # inputs
            inlinks=["x"],
            boot_memories=["h_boot"],
            step_net="stepnet",
            # outputs
            outlinks=[output],
            step_scopes="step_scopes",
            # attributes
            inlink_alias=["x@alias"],
            outlink_alias=[output_alias],
            pre_memories=[prememory],
            memories=[memory])

        ctx = core.DeviceContext.cpu_context()
        rnnop.infer_shape(self.scope)
        rnnop.run(self.scope, ctx)

    def test_recurrent(self):
        self.init()


if __name__ == '__main__':
    unittest.main()
