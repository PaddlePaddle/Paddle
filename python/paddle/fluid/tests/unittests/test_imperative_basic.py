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
from paddle.fluid.imperative.nn import FC
from test_imperative_base import new_program_scope


class MyLayer(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(MyLayer, self).__init__(name_scope)

    def forward(self, inputs):
        x = fluid.layers.relu(inputs)
        self._x_for_debug = x
        x = fluid.layers.elementwise_mul(x, x)
        x = fluid.layers.reduce_sum(x)
        return [x]


class MyPyLayer(fluid.imperative.PyLayer):
    def __init__(self):
        super(MyPyLayer, self).__init__()

    @staticmethod
    def forward(inputs):
        return np.tanh(inputs[0])

    @staticmethod
    def backward(inputs):
        inp, out, dout = inputs
        return np.array(dout) * (1 - np.square(np.array(out)))


class MLP(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(MLP, self).__init__(name_scope)
        self._fc1 = FC(self.full_name(),
                       3,
                       fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))
        self._fc2 = FC(self.full_name(),
                       4,
                       fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = self._fc2(x)
        x = fluid.layers.reduce_sum(x)
        return x


class SimpleRNNCell(fluid.imperative.Layer):
    def __init__(self, name_scope, step_input_size, hidden_size, output_size,
                 param_attr):
        super(SimpleRNNCell, self).__init__(name_scope)
        self.step_input_size = step_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._dype = core.VarDesc.VarType.FP32
        from paddle.fluid.layer_helper import LayerHelper
        self._helper = LayerHelper(
            'SimpleRNNCell', act="tanh", param_attr=param_attr)

    def _build_once(self, inputs, pre_hidden):
        i2h_param_shape = [self.step_input_size, self.hidden_size]
        h2h_param_shape = [self.hidden_size, self.hidden_size]
        h2o_param_shape = [self.output_size, self.hidden_size]
        self._i2h_w = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=i2h_param_shape,
            dtype=self._dtype,
            is_bias=False)
        self._h2h_w = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=h2h_param_shape,
            dtype=self._dtype,
            is_bias=False)
        self._h2o_w = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=h2o_param_shape,
            dtype=self._dtype,
            is_bias=False)

    def forward(self, input, pre_hidden):

        tmp_i2h = self._helper.create_variable_for_type_inference(self._dtype)
        tmp_h2h = self._helper.create_variable_for_type_inference(self._dtype)
        hidden = self._helper.create_variable_for_type_inference(self._dype)
        out = self._helper.create_variable_for_type_inference(self._dype)
        softmax_out = self._helper.create_variable_for_type_inference(
            self._dtype)
        reduce_out = self._helper.create_variable_for_type_inference(
            self._dtype)
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
        hidden = self._helper.append_activation(hidden)

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
            attrs={'dim': None,
                   'keep_dim': False,
                   'reduce_all': True})

        return reduce_out, hidden


class SimpleRNN(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(SimpleRNN, self).__init__(name_scope)
        self.seq_len = 4
        self._cell = SimpleRNNCell(
            self.full_name(),
            3,
            3,
            3,
            fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        outs = list()
        pre_hiddens = list()

        init_hidden = fluid.layers.tensor.create_parameter(
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
    def test_sum_op(self):
        x = np.ones([2, 2], np.float32)
        with fluid.imperative.guard():
            inputs = []
            for _ in range(10):
                inputs.append(fluid.imperative.base.to_variable(x))
            ret = fluid.layers.sums(inputs)
            loss = fluid.layers.reduce_sum(ret)
            loss._backward()
            self.assertTrue(np.allclose(ret._numpy(), x * 10))
            self.assertTrue(np.allclose(inputs[0]._gradient(), x))

    def test_layer(self):
        with fluid.imperative.guard():
            cl = core.Layer()
            cl.forward([])
            l = fluid.imperative.Layer("l")
            self.assertRaises(NotImplementedError, l.forward, [])

    def test_pylayer_func_id(self):

        with fluid.imperative.guard():

            class PyLayer1(fluid.imperative.PyLayer):
                def __init__(self):
                    super(PyLayer1, self).__init__()

                @staticmethod
                def forward(input):
                    return input

                @staticmethod
                def backward(input):
                    return input

            class PyLayer2(fluid.imperative.PyLayer):
                def __init__(self):
                    super(PyLayer2, self).__init__()

                @staticmethod
                def forward(input):
                    return input

                @staticmethod
                def backward(input):
                    return input

            py_layer_1 = PyLayer1()
            py_layer_2 = PyLayer2()
            py_layer_1(fluid.imperative.base.to_variable(np.ones([2, 2])))
            py_layer_2(fluid.imperative.base.to_variable(np.ones([2, 2])))
            id = py_layer_1.forward_id
            self.assertGreater(id, 0)
            self.assertEqual(py_layer_1.backward_id, id + 1)
            self.assertEqual(py_layer_2.forward_id, id + 2)
            self.assertEqual(py_layer_2.backward_id, id + 3)
            py_layer_1(fluid.imperative.base.to_variable(np.ones([2, 2])))
            self.assertEqual(py_layer_1.forward_id, id)

    def test_pylayer(self):
        np_inp = np.ones([2, 2], np.float32)
        with fluid.imperative.guard():
            my_py_layer = MyPyLayer()
            var_inp = fluid.imperative.base.to_variable(np_inp)
            outs = my_py_layer(var_inp)
            dy_out = np.sum(outs[0]._numpy())
            outs[0]._backward()
            dy_grad = var_inp._gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            # TODO(panyx0718): Paddle doesn't diff against data `inp`.
            x1 = inp * 1
            # TODO(panyx0718): If reduce_sum is skipped, the result is wrong.
            x = fluid.layers.reduce_sum(fluid.layers.tanh(x1))
            param_grads = fluid.backward.append_backward(
                x, parameter_list=[x1.name])[0]
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))

            static_out, static_grad = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[x.name, param_grads[1].name])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad, static_grad))

    def test_layer_in_out(self):
        np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)
        with fluid.imperative.guard():
            var_inp = fluid.imperative.base.to_variable(np_inp)
            l = MyLayer("my_layer")
            x = l(var_inp)[0]
            self.assertIsNotNone(x)
            dy_out = x._numpy()
            x._backward()
            dy_grad = l._x_for_debug._gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[3], append_batch_size=False)
            l = MyLayer("my_layer")
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

    def test_mlp(self):
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with fluid.imperative.guard():
            var_inp = fluid.imperative.base.to_variable(np_inp)
            mlp = MLP("mlp")
            out = mlp(var_inp)
            dy_out = out._numpy()
            out._backward()
            dy_grad = mlp._fc1._w._gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            mlp = MLP("mlp")
            out = mlp(inp)
            param_grads = fluid.backward.append_backward(
                out, parameter_list=[mlp._fc1._w.name])[0]
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(fluid.default_startup_program())

            static_out, static_grad = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[out.name, param_grads[1].name])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad, static_grad))

        params = mlp.parameters(True)
        self.assertEqual("mlp/MLP_0/FC_0_0.w_0", params[0].name)
        self.assertEqual("mlp/MLP_0/FC_0_0.b_0", params[1].name)
        self.assertEqual("mlp/MLP_0/FC_1_0.w_0", params[2].name)
        self.assertEqual("mlp/MLP_0/FC_1_0.b_0", params[3].name)
        self.assertEqual(len(params), 4)

        sublayers = mlp.sublayers(True)
        self.assertEqual(mlp._fc1, sublayers[0])
        self.assertEqual(mlp._fc2, sublayers[1])
        self.assertEqual(len(sublayers), 2)

    def test_rnn(self):
        np_inp = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                           [10.0, 11.0, 12.0]])
        np_inp = np_inp.reshape((1, 4, 3))
        np_inp = np_inp.astype(np.float32)
        with fluid.imperative.guard():
            var_inp = fluid.imperative.base.to_variable(np_inp)
            var_inp = fluid.layers.reshape(var_inp, shape=[1, 4, 3])
            simple_rnn = SimpleRNN("simple_rnn")
            outs, pre_hiddens = simple_rnn.forward(var_inp)
            dy_out = outs[3]._numpy()
            outs[3]._backward()
            dy_grad_h2o = simple_rnn._cell._h2o_w._gradient()
            dy_grad_h2h = simple_rnn._cell._h2h_w._gradient()
            dy_grad_i2h = simple_rnn._cell._i2h_w._gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[1, 4, 3], append_batch_size=False)
            simple_rnn = SimpleRNN("simple_rnn")
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


if __name__ == '__main__':
    unittest.main()
