import logging
import paddle.v2.framework.core as core
import unittest
import numpy as np
from paddle.v2.framework.op import Operator, CondOp


class PySimpleCond(object):
    '''
    A simple implementation of dynamic if-else based on numpy
    '''

    def __init__(self):
        array = [1] * 10
        for i in range(1, 10, 2):
            array[i] = 0
        self.cond = np.array(array)
        self.x = np.ones(shape=(10, 1)).astype("float32")

    def forward(self):
        self.index_t = np.where(self.cond == 1)
        self.index_f = np.where(self.cond == 0)
        y_t = self.x[self.index_t]
        y_f = self.x[self.index_f]
        y_t = y_t * 2.
        y_f = y_f * (-2.)
        output = np.zeros(shape=(10, 1))
        output[self.index_t] = y_t
        output[self.index_f] = y_f
        return output


class PySimpleCondTest(unittest.TestCase):
    def setUp(self):
        self.condnn = PySimpleCond()

    def test_forward(self):
        output = self.condnn.forward()


def create_tensor(scope, name, shape, np_data):
    tensor = scope.var(name).get_tensor()
    tensor.set_dims(shape)
    tensor.set(np_data, core.CPUPlace())
    return tensor


class TestCondOp(unittest.TestCase):
    '''
    Test CondOp

    equation:
        cond = [True, False, True, False, ...]
        y[index_t] = x[index_t] * 2.
        y[index_f] = x[index_f] * -2.
    outputs:
        y
    '''

    def setUp(self):
        self.py_cond = PySimpleCond()

    def forward(self):
        self.scope = core.Scope()
        self.create_global_variables()
        self.create_cond_op()
        self.create_sub_net()
        ctx = core.DeviceContext.create(core.CPUPlace())
        self.condop.run(self.scope, ctx)
        return np.array(self.scope.find_var("Out").get_tensor())

    def create_global_variables(self):
        x_np_data = self.py_cond.x
        create_tensor(self.scope, "X", [10, 1], x_np_data)
        cond_np_data = self.py_cond.cond.astype("int32")
        create_tensor(self.scope, "cond", [10, 1], cond_np_data)
        self.scope.var("SubScopes")
        self.scope.var("IndexTensors")
        self.scope.var("Out")

    def create_cond_op(self):
        self.condop = CondOp(
            Cond="cond",
            Xs=["X"],
            Outs=["Out"],
            SubScopes="SubScopes",
            IndexTensors="IndexTensors")

    def create_sub_net(self):
        truenet = core.Net.create()
        scale_op_t = Operator("scale", X='X', Out='Out', scale=2.)
        truenet.append_op(scale_op_t)
        truenet.complete_add_op(True)
        self.condop.set_truenet(truenet)

        falsenet = core.Net.create()
        scale_op_t = Operator("scale", X='X', Out='Out', scale=-2.)
        falsenet.append_op(scale_op_t)
        falsenet.complete_add_op(True)
        self.condop.set_falsenet(falsenet)

    def test_forward(self):
        print 'test cond op forward'
        pd_output = self.forward()
        py_output = self.py_cond.forward()
        print 'pd_output', pd_output
        print
        print 'py_output', py_output
        self.assertEqual(pd_output.shape, py_output.shape)
        print 'test passed'
        return 0


if __name__ == "__main__":
    exit(
        0
    )  # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957
    unittest.main()
