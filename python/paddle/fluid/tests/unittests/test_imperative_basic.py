# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import Linear
from test_imperative_base import new_program_scope


class MyLayer(fluid.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def forward(self, inputs):
        x = fluid.layers.relu(inputs)
        self._x_for_debug = x
        x = fluid.layers.elementwise_mul(x, x)
        x = fluid.layers.reduce_sum(x)
        return [x]


class MLP(fluid.Layer):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self._linear1 = None
        self._linear1 = Linear(
            input_size,
            3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)))
        self._linear2 = Linear(
            3,
            4,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        x = self._linear1(inputs)
        x = self._linear2(x)
        x = fluid.layers.reduce_sum(x)
        return x


class SimpleRNNCell(fluid.Layer):
    def __init__(self, step_input_size, hidden_size, output_size, param_attr):
        super(SimpleRNNCell, self).__init__()
        self.step_input_size = step_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._dtype = core.VarDesc.VarType.FP32
        self.param_attr = param_attr

        i2h_param_shape = [self.step_input_size, self.hidden_size]
        h2h_param_shape = [self.hidden_size, self.hidden_size]
        h2o_param_shape = [self.output_size, self.hidden_size]
        self._i2h_w = None
        self._i2h_w = self.create_parameter(
            attr=self.param_attr,
            shape=i2h_param_shape,
            dtype=self._dtype,
            is_bias=False)
        self._h2h_w = self.create_parameter(
            attr=self.param_attr,
            shape=h2h_param_shape,
            dtype=self._dtype,
            is_bias=False)
        self._h2o_w = self.create_parameter(
            attr=self.param_attr,
            shape=h2o_param_shape,
            dtype=self._dtype,
            is_bias=False)

    def forward(self, input, pre_hidden):
        tmp_i2h = self.create_variable(dtype=self._dtype)
        tmp_h2h = self.create_variable(dtype=self._dtype)
        hidden = self.create_variable(dtype=self._dtype)
        out = self.create_variable(dtype=self._dtype)
        softmax_out = self.create_variable(dtype=self._dtype)
        reduce_out = self.create_variable(dtype=self._dtype)
        self._helper.append_op(
            type="mul",
            inputs={"X": input,
                    "Y": self._i2h_w},
            outputs={"Out": tmp_i2h},
            attrs={"x_num_col_dims": 1,
                   "y_num_col_dims": 1})

        self._helper.append_op(
            type="mul",
            inputs={"X": pre_hidden,
                    "Y": self._h2h_w},
            outputs={"Out": tmp_h2h},
            attrs={"x_num_col_dims": 1,
                   "y_num_col_dims": 1})

        self._helper.append_op(
            type="elementwise_add",
            inputs={'X': tmp_h2h,
                    'Y': tmp_i2h},
            outputs={'Out': hidden},
            attrs={'axis': -1,
                   'use_mkldnn': False})
        hidden = self._helper.append_activation(hidden, act='tanh')

        self._helper.append_op(
            type="mul",
            inputs={"X": hidden,
                    "Y": self._h2o_w},
            outputs={"Out": out},
            attrs={"x_num_col_dims": 1,
                   "y_num_col_dims": 1})

        self._helper.append_op(
            type="softmax",
            inputs={"X": out},
            outputs={"Out": softmax_out},
            attrs={"use_cudnn": False})

        self._helper.append_op(
            type='reduce_sum',
            inputs={'X': softmax_out},
            outputs={'Out': reduce_out},
            attrs={'keep_dim': False,
                   'reduce_all': True})

        return reduce_out, hidden


class SimpleRNN(fluid.Layer):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.seq_len = 4
        self._cell = SimpleRNNCell(
            3,
            3,
            3,
            fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        outs = list()
        pre_hiddens = list()

        init_hidden = self.create_parameter(
            attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)),
            shape=[1, 3],
            dtype='float32',
            is_bias=False)
        pre_hidden = init_hidden
        for i in range(self.seq_len):
            input = fluid.layers.slice(
                inputs, axes=[1], starts=[i], ends=[i + 1])
            input = fluid.layers.reshape(input, shape=[1, 3])
            out_softmax, pre_hidden = self._cell(input, pre_hidden)
            outs.append(out_softmax)

        return outs, pre_hiddens


class TestImperative(unittest.TestCase):
    def test_isinstance(self):
        var = fluid.layers.data(shape=[1], name='x', dtype='float32')
        self.assertTrue(isinstance(var, fluid.Variable))
        with fluid.dygraph.guard():
            var_base = fluid.dygraph.base.to_variable(np.array([3, 4, 5]))
            self.assertTrue(isinstance(var_base, core.VarBase))
            self.assertTrue(isinstance(var_base, fluid.Variable))

    def test_create_VarBase(self):
        x = np.ones([2, 2], np.float32)
        y = np.zeros([3, 3], np.float32)
        with fluid.dygraph.guard():
            tmp = fluid.core.VarBase(value=x, place=fluid.core.CPUPlace())
            tmp2 = fluid.core.VarBase(y, fluid.core.CPUPlace())
            tmp3 = fluid.dygraph.base.to_variable(x)
            tmp4 = fluid.core.VarBase(y)
            tmp5 = fluid.core.VarBase(value=x)

            self.assertTrue(np.array_equal(x, tmp.numpy()))
            self.assertTrue(np.array_equal(y, tmp2.numpy()))
            self.assertTrue(np.array_equal(x, tmp3.numpy()))
            self.assertTrue(np.array_equal(y, tmp4.numpy()))
            self.assertTrue(np.array_equal(x, tmp5.numpy()))

    def test_sum_op(self):
        x = np.ones([2, 2], np.float32)
        with fluid.dygraph.guard():
            inputs = []
            for _ in range(10):
                tmp = fluid.dygraph.base.to_variable(x)
                tmp.stop_gradient = False
                inputs.append(tmp)
            ret = fluid.layers.sums(inputs)
            loss = fluid.layers.reduce_sum(ret)
            loss.backward()
        with fluid.dygraph.guard():
            inputs2 = []
            for _ in range(10):
                tmp = fluid.dygraph.base.to_variable(x)
                tmp.stop_gradient = False
                inputs2.append(tmp)
            ret2 = fluid.layers.sums(inputs2)
            loss2 = fluid.layers.reduce_sum(ret2)
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            loss2.backward(backward_strategy)

            self.assertTrue(np.allclose(ret.numpy(), x * 10))
            self.assertTrue(np.allclose(inputs[0].gradient(), x))
            self.assertTrue(np.allclose(ret2.numpy(), x * 10))
            a = inputs2[0].gradient()
            self.assertTrue(np.allclose(inputs2[0].gradient(), x))

    def test_empty_var(self):
        with fluid.dygraph.guard():
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(
                name="X", shape=[-1, 23, 48], dtype='float32')
            try:
                new_variable.numpy()
            except Exception as e:
                assert type(e) == core.EnforceNotMet

            try:
                new_variable.backward()
            except Exception as e:
                assert type(e) == core.EnforceNotMet

            try:
                new_variable.clear_gradient()
            except Exception as e:
                assert type(e) == core.EnforceNotMet

    def test_empty_grad(self):
        with fluid.dygraph.guard():
            x = np.ones([2, 2], np.float32)
            new_var = fluid.dygraph.base.to_variable(x)
            try:
                new_var.gradient()
            except Exception as e:
                assert type(e) == ValueError

            try:
                new_var.clear_gradient()
            except Exception as e:
                assert type(e) == core.EnforceNotMet

        with fluid.dygraph.guard():
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(
                name="X", shape=[-1, 23, 48], dtype='float32')
            try:
                new_variable.gradient()
            except Exception as e:
                assert type(e) == ValueError

    def test_set_persistable(self):
        with fluid.dygraph.guard():
            x = np.ones([2, 2], np.float32)
            new_var = fluid.dygraph.base.to_variable(x)
            self.assertFalse(new_var.persistable)
            new_var.persistable = True
            self.assertTrue(new_var.persistable)

    def test_layer(self):
        with fluid.dygraph.guard():
            cl = core.Layer()
            cl.forward([])
            l = fluid.Layer("l")
            self.assertRaises(NotImplementedError, l.forward, [])

    def test_layer_in_out(self):
        np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)
        with fluid.dygraph.guard():
            var_inp = fluid.dygraph.base.to_variable(np_inp)
            var_inp.stop_gradient = False
            l = MyLayer()
            print(var_inp)
            x = l(var_inp)[0]
            self.assertIsNotNone(x)
            dy_out = x.numpy()
            x.backward()
            dy_grad = l._x_for_debug.gradient()

        with fluid.dygraph.guard():
            var_inp2 = fluid.dygraph.base.to_variable(np_inp)
            var_inp2.stop_gradient = False
            l2 = MyLayer()
            x2 = l2(var_inp2)[0]
            self.assertIsNotNone(x2)
            dy_out2 = x2.numpy()
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            x2.backward(backward_strategy)
            dy_grad2 = l2._x_for_debug.gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[3], append_batch_size=False)
            l = MyLayer()
            x = l(inp)[0]
            param_grads = fluid.backward.append_backward(
                x, parameter_list=[l._x_for_debug.name])[0]
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))

            static_out, static_grad = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[x.name, param_grads[1].name])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad, static_grad))
        self.assertTrue(np.allclose(dy_out2, static_out))
        self.assertTrue(np.allclose(dy_grad2, static_grad))

    def test_mlp(self):
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with fluid.dygraph.guard():
            var_inp = fluid.dygraph.base.to_variable(np_inp)
            mlp = MLP(input_size=2)
            out = mlp(var_inp)
            dy_out = out.numpy()
            out.backward()
            dy_grad = mlp._linear1.weight.gradient()

        with fluid.dygraph.guard():
            var_inp2 = fluid.dygraph.base.to_variable(np_inp)
            mlp2 = MLP(input_size=2)
            out2 = mlp2(var_inp2)
            dy_out2 = out2.numpy()
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            out2.backward(backward_strategy)
            dy_grad2 = mlp2._linear1.weight.gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            mlp = MLP(input_size=2)
            out = mlp(inp)
            param_grads = fluid.backward.append_backward(
                out, parameter_list=[mlp._linear1.weight.name])[0]
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(fluid.default_startup_program())

            static_out, static_grad = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[out.name, param_grads[1].name])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad, static_grad))
        self.assertTrue(np.allclose(dy_out2, static_out))
        self.assertTrue(np.allclose(dy_grad2, static_grad))

        params = mlp.parameters(True)
        self.assertEqual("linear_0.w_0", params[0].name)
        self.assertEqual("linear_0.b_0", params[1].name)
        self.assertEqual("linear_1.w_0", params[2].name)
        self.assertEqual("linear_1.b_0", params[3].name)
        self.assertEqual(len(params), 4)

        sublayers = mlp.sublayers(True)
        self.assertEqual(mlp._linear1, sublayers[0])
        self.assertEqual(mlp._linear2, sublayers[1])
        self.assertEqual(len(sublayers), 2)

    def test_dygraph_vs_static(self):
        np_inp1 = np.random.rand(4, 3, 3)
        np_inp2 = np.random.rand(4, 3, 3)

        # dynamic graph
        with fluid.dygraph.guard():
            inp1 = fluid.dygraph.to_variable(np_inp1)
            inp2 = fluid.dygraph.to_variable(np_inp2)
            if np.sum(np_inp1) < np.sum(np_inp2):
                x = fluid.layers.elementwise_add(inp1, inp2)
            else:
                x = fluid.layers.elementwise_sub(inp1, inp2)
            dygraph_result = x.numpy()

        # static graph
        with new_program_scope():
            inp_data1 = fluid.layers.data(
                name='inp1', shape=[3, 3], dtype=np.float32)
            inp_data2 = fluid.layers.data(
                name='inp2', shape=[3, 3], dtype=np.float32)

            a = fluid.layers.expand(
                fluid.layers.reshape(
                    fluid.layers.reduce_sum(inp_data1), [1, 1]), [4, 1])
            b = fluid.layers.expand(
                fluid.layers.reshape(
                    fluid.layers.reduce_sum(inp_data2), [1, 1]), [4, 1])
            cond = fluid.layers.less_than(x=a, y=b)

            ie = fluid.layers.IfElse(cond)
            with ie.true_block():
                d1 = ie.input(inp_data1)
                d2 = ie.input(inp_data2)
                d3 = fluid.layers.elementwise_add(d1, d2)
                ie.output(d3)

            with ie.false_block():
                d1 = ie.input(inp_data1)
                d2 = ie.input(inp_data2)
                d3 = fluid.layers.elementwise_sub(d1, d2)
                ie.output(d3)
            out = ie()

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            static_result = exe.run(fluid.default_main_program(),
                                    feed={'inp1': np_inp1,
                                          'inp2': np_inp2},
                                    fetch_list=out)[0]
        self.assertTrue(np.allclose(dygraph_result, static_result))

    def test_rnn(self):
        np_inp = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                           [10.0, 11.0, 12.0]])
        np_inp = np_inp.reshape((1, 4, 3))
        np_inp = np_inp.astype(np.float32)
        with fluid.dygraph.guard():
            var_inp = fluid.dygraph.base.to_variable(np_inp)
            var_inp = fluid.layers.reshape(var_inp, shape=[1, 4, 3])
            simple_rnn = SimpleRNN()
            outs, pre_hiddens = simple_rnn.forward(var_inp)
            dy_out = outs[3].numpy()
            outs[3].backward()
            dy_grad_h2o = simple_rnn._cell._h2o_w.gradient()
            dy_grad_h2h = simple_rnn._cell._h2h_w.gradient()
            dy_grad_i2h = simple_rnn._cell._i2h_w.gradient()

        with fluid.dygraph.guard():
            var_inp2 = fluid.dygraph.base.to_variable(np_inp)
            var_inp2 = fluid.layers.reshape(var_inp2, shape=[1, 4, 3])
            simple_rnn2 = SimpleRNN()
            outs2, pre_hiddens2 = simple_rnn2.forward(var_inp2)
            dy_out2 = outs2[3].numpy()
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            outs2[3].backward(backward_strategy)
            dy_grad_h2o2 = simple_rnn2._cell._h2o_w.gradient()
            dy_grad_h2h2 = simple_rnn2._cell._h2h_w.gradient()
            dy_grad_i2h2 = simple_rnn2._cell._i2h_w.gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[1, 4, 3], append_batch_size=False)
            simple_rnn = SimpleRNN()
            outs, pre_hiddens = simple_rnn(inp)
            param_grads = fluid.backward.append_backward(outs[3])
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            static_out, static_grad_h2o, static_grad_h2h, static_grad_i2h = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[
                    outs[3].name, param_grads[0][1].name,
                    param_grads[1][1].name, param_grads[2][1].name
                ])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad_h2o, static_grad_h2o))
        self.assertTrue(np.allclose(dy_grad_h2h, static_grad_h2h))
        self.assertTrue(np.allclose(dy_grad_i2h, static_grad_i2h))
        self.assertTrue(np.allclose(dy_out2, static_out))
        self.assertTrue(np.allclose(dy_grad_h2o2, static_grad_h2o))
        self.assertTrue(np.allclose(dy_grad_h2h2, static_grad_h2h))
        self.assertTrue(np.allclose(dy_grad_i2h2, static_grad_i2h))

    def test_layer_attrs(self):
        layer = fluid.dygraph.Layer("test")
        layer.test_attr = 1
        self.assertFalse(hasattr(layer, "whatever"))
        self.assertTrue(hasattr(layer, "test_attr"))
        self.assertEqual(layer.test_attr, 1)

        my_layer = MyLayer()
        my_layer.w1 = my_layer.create_parameter([3, 3])
        my_layer.add_parameter('w2', None)
        self.assertEqual(len(my_layer.parameters()), 1)
        self.assertRaises(TypeError, my_layer.__setattr__, 'w1', 'str')
        my_layer.w1 = None
        self.assertEqual(len(my_layer.parameters()), 0)
        my_layer.l1 = fluid.dygraph.Linear(3, 3)
        self.assertEqual(len(my_layer.sublayers()), 1)
        self.assertRaises(TypeError, my_layer.__setattr__, 'l1', 'str')
        my_layer.l1 = None
        self.assertEqual(len(my_layer.sublayers()), 0)


if __name__ == '__main__':
    unittest.main()
