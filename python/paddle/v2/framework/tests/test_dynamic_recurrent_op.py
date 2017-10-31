import logging
import paddle.v2.framework.core as core
import unittest
import copy
import math
from paddle.v2.framework.op import Operator, DynamicRecurrentOp
import numpy as np
from op_test import get_backward_op

lod_py = [[0, 1, 3, 6]]
input_dim = 6
num_sents = len(lod_py[0]) - 1
num_instances = lod_py[0][-1]
weight_dim = 10


def py_sigmoid(x):
    return 1. / (1. + np.exp(-x))


def create_array(shape, data=None, is_random=False):
    if is_random:
        return np.random.normal(size=shape).astype("float32")

    if data:
        arr = np.array([data
                        for i in range(shape[0] * shape[1])]).astype("float32")
    else:
        arr = np.array([0.1 * i
                        for i in range(shape[0] * shape[1])]).astype("float32")
    return arr.reshape(shape)


def create_tensor(scope, name, np_data):
    tensor = scope.var(name).get_tensor()
    tensor.set_dims(np_data.shape)
    tensor.set(np_data, core.CPUPlace())
    return tensor


def create_raw_tensor(np_data, place):
    tensor = core.LoDTensor()
    tensor.set(np_data, place)
    return tensor


class PyRNN(object):
    def __init__(self):
        self.x = create_array((lod_py[0][-1], input_dim), is_random=True)
        self.W = create_array((input_dim, input_dim), data=1., is_random=True)
        self.U = create_array((input_dim, input_dim), 1., is_random=True)
        self.h_boot = create_array((num_sents, input_dim), 1., is_random=True)

        self.num_steps = len(lod_py[0]) - 1
        self.place = core.CPUPlace()
        self.mems = [None for i in range(3)]
        self.tmp_outputs = {
            'x': [],
            'h@pre': [],
            'xW': [],
            'hU': [],
            'sum': [],
            'h@state': [],
        }

    def __call__(self):
        tensor = core.LoDTensor(lod_py)
        tensor.set(self.x, self.place)

        self.input_ta = core.TensorArray()
        self.meta = self.input_ta.unpack(tensor, 0, True)
        self.output_ta = core.TensorArray()

        for step in range(3):
            input = self.get_step_input(step)
            pre_state = self.get_pre_state(step)
            out = self._run_step(input, pre_state)
            self.mems[step] = out

    def _run_step(self, x, pre_state):
        xW = np.matmul(x, self.W).astype('float32')
        hU = np.matmul(pre_state, self.U).astype('float32')

        sum = xW + hU
        out = py_sigmoid(sum)

        self.tmp_outputs['x'].append(x)
        self.tmp_outputs['h@pre'].append(pre_state)
        self.tmp_outputs['xW'].append(xW)
        self.tmp_outputs['hU'].append(hU)
        self.tmp_outputs['sum'].append(sum)
        self.tmp_outputs['h@state'].append(out)
        return out

    def run_core_step(self, x, pre_state):
        '''
        run a paddle step
        '''
        # create global variables
        scope = core.Scope()
        # prepare inputs for stepnet
        create_tensor(scope, "x", x)
        create_tensor(scope, "h@pre", pre_state)

        create_tensor(scope, "W", self.W)
        create_tensor(scope, "U", self.U)

        scope.var("h@state")
        scope.var("xW")
        scope.var("hU")
        scope.var("sum")

        # create a net op
        step_unit = core.Net.create()
        x_fc_op = Operator("mul", X="x", Y="W", Out="xW")
        h_fc_op = Operator("mul", X="h@pre", Y="U", Out="hU")
        sum_op = Operator("sum", X=["xW", "hU"], Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@state")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            step_unit.append_op(op)
        step_unit.complete_add_op(True)

        # run this step
        ctx = core.DeviceContext.create(core.CPUPlace())
        step_unit.run(scope, ctx)

        res = {}
        for var in ['xW', 'hU', 'sum', 'h@state']:
            tensor = np.array(scope.var(var).get_tensor())
            res[var] = tensor
        return res

    def get_step_input(self, step):
        return np.array(self.input_ta.read(step))

    def get_pre_state(self, step):
        if step == 0:
            # process boot memory
            h_boot = copy.copy(self.h_boot)
            h_boot[0] = self.h_boot[2]
            h_boot[1] = self.h_boot[1]
            h_boot[2] = self.h_boot[0]
            return h_boot
        else:
            # process the previous memory
            pre_mem = np.array(self.mems[step - 1])
            x = self.get_step_input(step)
            num_instances = x.shape[0]
            pre_mem = pre_mem[:num_instances]
            return pre_mem


class PyRNNTest(unittest.TestCase):
    py = PyRNN()

    def setUp(self):
        self.py()

    def test_one_step(self):
        print '=' * 50
        print 'one step'

        py_tmp_outputs = self.py.tmp_outputs
        for step in range(3):
            x = self.py.get_step_input(step)
            pre_state = self.py.get_pre_state(step)
            res = self.py.run_core_step(x, pre_state)
            for var in ['xW', 'hU', 'sum', 'h@state']:
                py_data = py_tmp_outputs[var][step]
                core_data = res[var]
                print '<' * 10
                print var
                print py_data
                print core_data
                self.assertTrue(np.isclose(py_data, core_data).all())
        print '-' * 50


class DynamicRecurrentOpTest(unittest.TestCase):
    py = PyRNN()

    def setUp(self):
        self.py()

        self.scope = core.Scope()
        self.ctx = core.DeviceContext.create(core.CPUPlace())

        self._create_global_variables()
        self._create_rnn_op()
        self._create_step_net()

        self.rnnop.run(self.scope, self.ctx)
        self.vars = ['x', 'xW', 'hU', 'h@pre', 'sum', 'h@state']

    def test_all_steps(self):
        print '=' * 50
        print 'test all steps'
        for step in range(3):
            for var in self.vars:
                tensor = self.rnnop.step_tensor(step, var)
                tensor = np.array(tensor)
                py_tensor = self.py.tmp_outputs[var][step]
                print '>' * 10
                print var
                print tensor
                print py_tensor
                self.assertTrue(np.isclose(tensor, py_tensor).all())

    def test_state_equals_pre_state(self):
        for step in range(1, 3):
            pre_state = np.array(self.rnnop.get_state('h@state').read(step - 1))
            state_pre = np.array(self.rnnop.get_pre_state('h@state').read(step))
            num_instances = min(pre_state.shape[0], state_pre.shape[0])
            pre_state = pre_state[:num_instances]
            state_pre = state_pre[:num_instances]
            self.assertTrue(np.isclose(pre_state, state_pre).all())

    def _create_rnn_op(self):
        self.rnnop = DynamicRecurrentOp(
            # inputs
            inputs=["x"],
            initial_states=["h_boot"],
            step_net="step_unit",
            # outputs
            # outputs=["h@state"],
            outputs=["xW"],
            step_scopes="step_scopes",
            # attributes
            ex_states=["h@pre"],
            states=["h@state"])

    def _create_step_net(self):
        step_unit = core.Net.create()
        x_fc_op = Operator("mul", X="x", Y="W", Out="xW")
        h_fc_op = Operator("mul", X="h@pre", Y="U", Out="hU")
        sum_op = Operator("sum", X=["xW", "hU"], Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@state")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            step_unit.append_op(op)
        step_unit.complete_add_op(True)
        self.rnnop.set_step_unit(step_unit)

    def _create_global_variables(self):
        x_tensor = create_tensor(self.scope, "x", self.py.x)
        x_tensor.set_lod(lod_py)
        create_tensor(self.scope, "W", self.py.W)
        create_tensor(self.scope, "U", self.py.U)
        create_tensor(self.scope, "h_boot", self.py.h_boot)
        self.scope.var("step_scopes")
        self.scope.var("h@state")
        self.scope.var("xW")


class PyRNNGrad(object):
    '''
    This is an independent gradient test for RNN.

    There are some reasonable hypothesis:

    1. RNN's output LoD equals inputs
    '''

    def __init__(self):
        self.place = core.CPUPlace()
        self.ctx = core.DeviceContext.create(core.CPUPlace())
        self.vars = ['h@state@GRAD', ]

        # reuse the PyRNN for the forward variables
        self.py = PyRNN()

        self.num_steps = len(lod_py[0]) - 1
        self.place = core.CPUPlace()
        self.tmp_outputs = {
            'x@GRAD': [],
            'h@pre@GRAD': [],
            'xW@GRAD': [],
            'hU@GRAD': [],
            'sum@GRAD': [],
            'h@state@GRAD': [],
            'W@GRAD': [],
            'U@GRAD': [],
        }
        # the backward variables
        self.h_grad = create_array((num_instances, input_dim), is_random=True)

        # split the inputs(grad)
        tensor = core.LoDTensor(lod_py)
        tensor.set(self.h_grad, self.place)
        # the grad(h@state@GRAD)
        self.input_ta = core.TensorArray()
        self.meta = self.input_ta.unpack(tensor, 0, True)

    def __call__(self):
        self.py()
        for step in range(3):
            self.run_step(step)
        return self.tmp_outputs

    def run_step(self, step):
        '''
        the inputs are output@grad
        the grads should be splitted by TensorArray
        '''
        netop = self._get_forward_netop()
        scope = self._init_step_scope(step)
        backop = self._gen_core_op(scope, netop)

        backop.run(scope, self.ctx)

        for var in self.tmp_outputs:
            self.tmp_outputs[var].append(np.array(scope.var(var).get_tensor()))

    def get_step_input(self, step):
        '''
        the step input, the h@state@GRAD
        '''
        return np.array(self.input_ta.read(step))

    def _init_step_scope(self, step):
        '''
        create the variables needed in step scope
        '''
        # init forward inputs
        scope = core.Scope()
        for var in 'W U h_boot'.split():
            data = getattr(self.py, var)
            create_tensor(scope, var, data)

        # init forward tmp outputs
        for var, datas in self.py.tmp_outputs.items():
            data = datas[step]
            create_tensor(scope, var, data)

        # set output@grad
        grad = self.get_step_input(step)
        scope.var('h@state@GRAD').get_tensor().set(grad, self.place)

        # create temp output vars
        for var in self.tmp_outputs.keys():
            scope.var(var)

        # # init W@GRAD and U@GRAD to zero because they are not just output
        # w_grad_data = create_array((input_dim, input_dim), data=0.)
        # scope.var('W@GRAD').get_tensor().set(w_grad_data, self.place)

        # u_grad_data = create_array((input_dim, input_dim), data=0.)
        # scope.var('U@GRAD').get_tensor().set(u_grad_data, self.place)

        return scope

    def _gen_core_op(self, scope, netop):
        '''
        input a NetOp that describes a forward network.
        '''
        backward_op = get_backward_op(scope, netop, {})
        for input in backward_op.input_vars():
            var = scope.var(input)
            var.get_tensor()
        for output in backward_op.output_vars():
            var = scope.var(input)
            var.get_tensor()
        return backward_op

    def _get_forward_netop(self):
        step_unit = core.Net.create()
        x_fc_op = Operator("mul", X="x", Y="W", Out="xW")
        h_fc_op = Operator("mul", X="h@pre", Y="U", Out="hU")
        sum_op = Operator("sum", X=["xW", "hU"], Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@state")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            step_unit.append_op(op)
        step_unit.complete_add_op(True)
        return step_unit


class PyRNNGradTest(unittest.TestCase):
    def setUp(self):
        self.py = PyRNNGrad()
        self.py()

    def test_forward(self):
        res = self.py()


class DynamicRecurrentGradientOpTest(unittest.TestCase):
    py = PyRNNGrad()

    def setUp(self):
        self.py_res = self.py()

        self._create_rnn_op()
        self._create_step_net()

        self.ctx = core.DeviceContext.create(core.CPUPlace())

    def test_all_steps(self):
        self.init_scope()
        self._create_rnn_op()

        self.rnnop.run(self.scope, self.ctx)
        print '!' * 50
        print 'backward', self.backward_rnnop
        self.backward_rnnop.run(self.scope, self.ctx)
        print '>' * 50

    def init_scope(self):
        # parent scope
        self.scope = core.Scope()
        x_tensor = create_tensor(self.scope, "x", self.py.py.x)
        x_tensor.set_lod(lod_py)
        create_tensor(self.scope, "W", self.py.py.W)
        create_tensor(self.scope, "U", self.py.py.U)
        create_tensor(self.scope, "h_boot", self.py.py.h_boot)
        self.scope.var("step_scopes")
        self.scope.var("step_scopes@GRAD")
        self.scope.var("x@GRAD")
        self.scope.var("h@state")
        self.scope.var("h_boot@GRAD")
        # TODO try some random
        tensor = create_tensor(
            self.scope,
            "h@state@GRAD",
            create_array(
                (num_instances, input_dim), is_random=True))
        tensor.set_lod(lod_py)
        create_tensor(self.scope, "W@GRAD",
                      create_array([input_dim, input_dim], 0.))
        create_tensor(self.scope, "U@GRAD",
                      create_array([input_dim, input_dim], 0.))

    def _create_rnn_op(self):
        self.rnnop = DynamicRecurrentOp(
            # inputs
            inputs=["x"],
            initial_states=["h_boot"],
            step_net="step_unit",
            # outputs
            # outputs=["h@state"],
            outputs=["h@state"],
            step_scopes="step_scopes",
            parameters=["W", "U"],
            # attributes
            ex_states=["h@pre"],
            states=["h@state"])

        self._create_step_net()
        self.backward_rnnop = self.rnnop.backward()
        print 'backward_rnnop', self.backward_rnnop

    def _create_step_net(self):
        '''
        create forward stepnet
        '''
        step_unit = core.Net.create()
        x_fc_op = Operator("mul", X="x", Y="W", Out="xW")
        h_fc_op = Operator("mul", X="h@pre", Y="U", Out="hU")
        sum_op = Operator("sum", X=["xW", "hU"], Out="sum")
        sig_op = Operator("sigmoid", X="sum", Y="h@state")

        for op in [x_fc_op, h_fc_op, sum_op, sig_op]:
            step_unit.append_op(op)
        step_unit.complete_add_op(True)

        self.rnnop.set_step_unit(step_unit)


if __name__ == '__main__':
    unittest.main()
