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

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import ParamBase, EagerParamBase
from paddle.jit import ProgramTranslator
from paddle.fluid.framework import _test_eager_guard, in_dygraph_mode


class L1(fluid.Layer):

    def __init__(self):
        super(L1, self).__init__()
        self._param_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.1))
        self.w1 = self.create_parameter(attr=self._param_attr,
                                        shape=[2, 2],
                                        dtype='float32',
                                        is_bias=False)
        self.w2 = self.create_parameter(attr=self._param_attr,
                                        shape=[2, 2],
                                        dtype='float32',
                                        is_bias=False)

    def forward(self):
        return self.w1 + self.w2


class L2(fluid.Layer):

    def __init__(self):
        super(L2, self).__init__()
        self.layer1 = L1()
        self.layer2 = L1()

    def forward(self):
        return self.layer1() + self.layer2()


class L3(fluid.Layer):

    def __init__(self):
        super(L3, self).__init__()
        self.layer1 = L2()
        self.layer2 = L2()

    def forward(self):
        return self.layer1() + self.layer2()


class TestBaseLayer(unittest.TestCase):

    def func_test_one_level(self):
        with fluid.dygraph.guard():
            l = L1()
            ret = l()
            expected_names = ['l1.w1', 'l1.w2']
            idx = 0
            for name, _ in l.named_parameters(prefix='l1'):
                self.assertEqual(name, expected_names[idx])
                idx += 1
            np.testing.assert_allclose(ret.numpy(),
                                       0.2 * np.ones([2, 2]),
                                       rtol=1e-05)

    def test_one_level(self):
        with _test_eager_guard():
            self.func_test_one_level()
        self.func_test_one_level()

    def func_test_three_level(self):
        with fluid.dygraph.guard():
            l = L3()
            expected_names = [
                'l3.layer1.layer1.w1',
                'l3.layer1.layer1.w2',
                'l3.layer1.layer2.w1',
                'l3.layer1.layer2.w2',
                'l3.layer2.layer1.w1',
                'l3.layer2.layer1.w2',
                'l3.layer2.layer2.w1',
                'l3.layer2.layer2.w2',
            ]
            idx = 0
            for name, _ in l.named_parameters(prefix='l3'):
                self.assertEqual(name, expected_names[idx])
                idx += 1
            ret = l()
            np.testing.assert_allclose(ret.numpy(),
                                       0.8 * np.ones([2, 2]),
                                       rtol=1e-05)

    def test_three_level(self):
        with _test_eager_guard():
            self.func_test_three_level()
        self.func_test_three_level()

    def func_test_add_parameter_with_error(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            param = net.create_parameter(shape=[1])

            with self.assertRaises(TypeError):
                net.add_parameter(10, param)

            with self.assertRaises(KeyError):
                net.add_parameter("param.name", param)

            with self.assertRaises(KeyError):
                net.add_parameter("", param)

            with self.assertRaises(KeyError):
                net.test_param = 10
                net.add_parameter("test_param", param)

            with self.assertRaises(TypeError):
                net.add_parameter("no_param", 10)

            load_param = net.create_parameter(shape=[1])
            net._loaddict_holder[load_param.name] = load_param
            net.add_parameter("load_param", load_param)

    def test_add_parameter_with_error(self):
        with _test_eager_guard():
            self.func_test_add_parameter_with_error()
        self.func_test_add_parameter_with_error()


class BufferLayer(fluid.Layer):

    def __init__(self):
        super(BufferLayer, self).__init__()
        buffer_var = to_variable(np.zeros([2, 4]).astype('int32'))
        self.register_buffer("layer_buffer", buffer_var)

    def forward(self):
        pass


class BufferNet(fluid.Layer):

    def __init__(self):
        super(BufferNet, self).__init__()
        self.buffer_layer = BufferLayer()
        self.w1 = self.create_parameter(shape=[2, 2],
                                        dtype='float32',
                                        is_bias=False)
        buffer_var = to_variable(np.ones([2, 4]).astype('int32'))
        self.register_buffer("net_buffer", buffer_var)

        self.new_buffer = to_variable(np.ones([4, 2]).astype('int32'))

    def forward(self):
        pass


class TestBuffer(unittest.TestCase):

    def func_test_buffers_and_named_buffers(self):

        def names(named_buffers):
            return [name for name, _ in named_buffers]

        with fluid.dygraph.guard():
            layer = BufferLayer()
            net = BufferNet()

            self.assertEqual(len(layer.buffers()), 1)
            self.assertEqual(names(layer.named_buffers()), ['layer_buffer'])

            self.assertEqual(len(net.buffers()), 3)
            self.assertEqual(
                names(net.named_buffers()),
                ['net_buffer', 'new_buffer', 'buffer_layer.layer_buffer'])

            self.assertEqual(len(net.buffers(include_sublayers=False)), 2)
            self.assertEqual(names(net.named_buffers(include_sublayers=False)),
                             ['net_buffer', 'new_buffer'])

    def test_buffers_and_named_buffers(self):
        with _test_eager_guard():
            self.func_test_buffers_and_named_buffers()
        self.func_test_buffers_and_named_buffers()

    def func_test_register_buffer_with_error(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var = to_variable(np.zeros([1]))

            with self.assertRaisesRegexp(TypeError,
                                         "name of buffer should be a string"):
                net.register_buffer(12, var)

            with self.assertRaisesRegexp(TypeError,
                                         "buffer should be a Paddle.Tensor"):
                if in_dygraph_mode():
                    net.register_buffer("buffer_name",
                                        EagerParamBase([2, 2], 'float32'))
                else:
                    net.register_buffer("buffer_name",
                                        ParamBase([2, 2], 'float32'))

            with self.assertRaisesRegexp(KeyError,
                                         "name of buffer can not contain"):
                net.register_buffer("buffer.name", var)

            with self.assertRaisesRegexp(KeyError,
                                         "name of buffer can not be empty"):
                net.register_buffer("", var)

            net.attr_name = 10
            with self.assertRaisesRegexp(KeyError, "already exists"):
                net.register_buffer("attr_name", var)

            del net.attr_name
            if in_dygraph_mode():
                net.attr_name = EagerParamBase([2, 2], 'float32')
            else:
                net.attr_name = ParamBase([2, 2], 'float32')
            with self.assertRaisesRegexp(KeyError, "already exists"):
                net.register_buffer("attr_name", var)

    def test_register_buffer_with_error(self):
        with _test_eager_guard():
            self.func_test_register_buffer_with_error()
        self.func_test_register_buffer_with_error()

    def func_test_register_buffer_same_name(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var1 = to_variable(np.zeros([1]))
            var2 = to_variable(np.zeros([2]))
            var3 = to_variable(np.zeros([3]))

            net.register_buffer("buffer_name", var1)
            self.assert_var_base_equal(net.buffer_name, var1)
            net.register_buffer("buffer_name", var2)
            self.assert_var_base_equal(net.buffer_name, var2)
            net.register_buffer("buffer_name", var3)
            self.assert_var_base_equal(net.buffer_name, var3)

    def test_register_buffer_same_name(self):
        with _test_eager_guard():
            self.func_test_register_buffer_same_name()
        self.func_test_register_buffer_same_name()

    def func_test_buffer_not_persistable(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var1 = to_variable(np.zeros([1]))

            net.register_buffer("buffer_name", var1, persistable=False)
            self.assertEqual(len(net.buffers()), 1)
            self.assertEqual(len(net.state_dict()), 0)

    def test_buffer_not_persistable(self):
        with _test_eager_guard():
            self.func_test_buffer_not_persistable()
        self.func_test_buffer_not_persistable()

    def func_test_buffer_not_persistable_del(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var1 = to_variable(np.zeros([1]))
            net.register_buffer("buffer_name", var1, persistable=False)
            del net.buffer_name
            self.assertEqual(len(net.buffers()), 0)

    def test_buffer_not_persistable_del(self):
        with _test_eager_guard():
            self.func_test_buffer_not_persistable_del()
        self.func_test_buffer_not_persistable_del()

    def func_test_buffer_not_persistable_overwrite(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var1 = to_variable(np.zeros([1]))
            var2 = to_variable(np.zeros([2]))
            net.register_buffer("buffer_name", var1, persistable=False)
            net.register_buffer("buffer_name", var2)

            # Allow to overwrite a non-persistable buffer with a persistable var.
            self.assertEqual(len(net.buffers()), 1)
            self.assertEqual(len(net.state_dict()), 1)

            net.register_buffer("buffer_name", var1, persistable=False)
            self.assertEqual(len(net.buffers()), 1)
            self.assertEqual(len(net.state_dict()), 0)

    def test_buffer_not_persistable_overwrite(self):
        with _test_eager_guard():
            self.func_test_buffer_not_persistable_overwrite()
        self.func_test_buffer_not_persistable_overwrite()

    def func_test_buffer_not_persistable_assign(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var1 = to_variable(np.zeros([1]))
            net.register_buffer("buffer_name", var1, persistable=False)

            # Assigning Nones will remove the buffer, but allow to re-assign
            # to remark it as buffer.
            net.buffer_name = None
            self.assertEqual(len(net.buffers()), 0)
            self.assertEqual(len(net.state_dict()), 0)

            net.buffer_name = var1
            self.assertEqual(len(net.buffers()), 1)
            self.assertEqual(len(net.state_dict()), 0)

            # Re-assign a ParamBase will remove the buffer.
            if in_dygraph_mode():
                net.buffer_name = EagerParamBase([2, 2], 'float32')
            else:
                net.buffer_name = ParamBase([2, 2], 'float32')
            self.assertEqual(len(net.buffers()), 0)
            self.assertEqual(len(net.state_dict()), 1)

    def test_buffer_not_persistable_assign(self):
        with _test_eager_guard():
            self.func_test_buffer_not_persistable_assign()
        self.func_test_buffer_not_persistable_assign()

    def func_test_buffer_not_persistable_load(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var1 = to_variable(np.zeros([1]))
            net.register_buffer("buffer_name", var1, persistable=False)
            net.load_dict({})

    def test_buffer_not_persistable_load(self):
        with _test_eager_guard():
            self.func_test_buffer_not_persistable_load()
        self.func_test_buffer_not_persistable_load()

    def func_test_buffer_state_dict(self):
        with fluid.dygraph.guard():
            net = fluid.Layer()
            var1 = to_variable(np.zeros([2, 3]))
            var2 = to_variable(np.zeros([3, 2]))
            net.register_buffer("buffer_var1", var1)
            net.register_buffer("buffer_var2", var2, persistable=False)

            self.assertEqual(len(net.state_dict()), 1)
            self.assertEqual([name for name, _ in net.state_dict().items()],
                             ["buffer_var1"])

            # load state_dict
            net_load = fluid.Layer()
            var = to_variable(np.ones([2, 3]))
            net_load.register_buffer("buffer_var1", var)
            net_load.load_dict(net.state_dict())

            self.assert_var_base_equal(net_load.buffer_var1, var1)

    def test_buffer_state_dict(self):
        with _test_eager_guard():
            self.func_test_buffer_state_dict()
        self.func_test_buffer_state_dict()

    def assert_var_base_equal(self, var1, var2):
        np.testing.assert_array_equal(var1.numpy(), var2.numpy())


class BufferNetWithModification(paddle.nn.Layer):

    def __init__(self, shape):
        super(BufferNetWithModification, self).__init__()

        self.buffer1 = paddle.zeros(shape, 'int32')
        self.buffer2 = paddle.zeros(shape, 'int32')

    @paddle.jit.to_static
    def forward(self, x):
        self.buffer1 += x
        self.buffer2 = self.buffer1 + x

        out = self.buffer1 + self.buffer2

        return out


class TestModifiedBuffer(unittest.TestCase):

    def funcsetUp(self):
        paddle.disable_static()
        self.prog_trans = ProgramTranslator()
        self.shape = [10, 16]

    def _run(self, to_static=False):
        self.prog_trans.enable(to_static)

        x = paddle.ones([1], 'int32')
        net = BufferNetWithModification(self.shape)
        out = net(x)

        return out, net.buffer1, net.buffer2

    def func_test_modified(self):
        self.funcsetUp()
        dy_outs = self._run(False)
        st_outs = self._run(True)

        for i in range(len(dy_outs)):
            np.testing.assert_array_equal(dy_outs[i].numpy(),
                                          st_outs[i].numpy())

    def test_modified(self):
        with _test_eager_guard():
            self.func_test_modified()
        self.func_test_modified()


class TestLayerTo(unittest.TestCase):

    def funcsetUp(self):
        paddle.disable_static()
        self.linear = paddle.nn.Linear(2, 2)
        self.new_grad = np.random.random([2, 2])
        self.linear.weight._set_grad_ivar(paddle.to_tensor(self.new_grad))
        buffer = paddle.to_tensor([0.0], dtype='float32')
        self.linear.register_buffer("buf_name", buffer, persistable=True)

        sublayer = paddle.nn.Conv1D(3, 2, 3)
        self.linear.add_sublayer("1", sublayer)

    def func_test_to_api(self):
        self.linear.to(dtype='double')
        self.assertEqual(self.linear.weight.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        self.assertEqual(self.linear.buf_name.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        np.testing.assert_allclose(self.linear.weight.grad.numpy(),
                                   self.new_grad,
                                   rtol=1e-05)
        self.assertEqual(self.linear.weight._grad_ivar().dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)

        self.linear.to()
        self.assertEqual(self.linear.weight.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        self.assertEqual(self.linear.buf_name.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        np.testing.assert_allclose(self.linear.weight.grad.numpy(),
                                   self.new_grad,
                                   rtol=1e-05)
        self.assertEqual(self.linear.weight._grad_ivar().dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        for p in self.linear.parameters():
            if in_dygraph_mode():
                self.assertTrue(
                    isinstance(p, paddle.fluid.framework.EagerParamBase))
            else:
                self.assertTrue(isinstance(p, paddle.fluid.framework.ParamBase))

        if paddle.fluid.is_compiled_with_cuda():
            self.linear.to(device=paddle.CUDAPlace(0))
            self.assertTrue(self.linear.weight.place.is_gpu_place())
            self.assertEqual(self.linear.weight.place.gpu_device_id(), 0)
            self.assertTrue(self.linear.buf_name.place.is_gpu_place())
            self.assertEqual(self.linear.buf_name.place.gpu_device_id(), 0)
            self.assertTrue(
                self.linear.weight._grad_ivar().place.is_gpu_place())
            self.assertEqual(
                self.linear.weight._grad_ivar().place.gpu_device_id(), 0)

            self.linear.to(device='gpu:0')
            self.assertTrue(self.linear.weight.place.is_gpu_place())
            self.assertEqual(self.linear.weight.place.gpu_device_id(), 0)
            self.assertTrue(self.linear.buf_name.place.is_gpu_place())
            self.assertEqual(self.linear.buf_name.place.gpu_device_id(), 0)
            self.assertTrue(
                self.linear.weight._grad_ivar().place.is_gpu_place())
            self.assertEqual(
                self.linear.weight._grad_ivar().place.gpu_device_id(), 0)
            for p in self.linear.parameters():
                if in_dygraph_mode():
                    self.assertTrue(
                        isinstance(p, paddle.fluid.framework.EagerParamBase))
                else:
                    self.assertTrue(
                        isinstance(p, paddle.fluid.framework.ParamBase))

        self.linear.to(device=paddle.CPUPlace())
        self.assertTrue(self.linear.weight.place.is_cpu_place())
        self.assertTrue(self.linear.buf_name.place.is_cpu_place())
        self.assertTrue(self.linear.weight._grad_ivar().place.is_cpu_place())

        self.linear.to(device='cpu')
        self.assertTrue(self.linear.weight.place.is_cpu_place())
        self.assertTrue(self.linear.buf_name.place.is_cpu_place())
        self.assertTrue(self.linear.weight._grad_ivar().place.is_cpu_place())

        self.assertRaises(ValueError, self.linear.to, device=1)

        self.assertRaises(AssertionError, self.linear.to, blocking=1)

    def func_test_to_api_paddle_dtype(self):
        self.linear.to(dtype=paddle.float64)
        self.assertEqual(self.linear.weight.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        self.assertEqual(self.linear.buf_name.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        np.testing.assert_allclose(self.linear.weight.grad.numpy(),
                                   self.new_grad,
                                   rtol=1e-05)
        self.assertEqual(self.linear.weight._grad_ivar().dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)

        self.linear.to()
        self.assertEqual(self.linear.weight.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        self.assertEqual(self.linear.buf_name.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        np.testing.assert_allclose(self.linear.weight.grad.numpy(),
                                   self.new_grad,
                                   rtol=1e-05)
        self.assertEqual(self.linear.weight._grad_ivar().dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        for p in self.linear.parameters():
            if in_dygraph_mode():
                self.assertTrue(
                    isinstance(p, paddle.fluid.framework.EagerParamBase))
            else:
                self.assertTrue(isinstance(p, paddle.fluid.framework.ParamBase))

    def func_test_to_api_numpy_dtype(self):
        self.linear.to(dtype=np.float64)
        self.assertEqual(self.linear.weight.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        self.assertEqual(self.linear.buf_name.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        np.testing.assert_allclose(self.linear.weight.grad.numpy(),
                                   self.new_grad,
                                   rtol=1e-05)
        self.assertEqual(self.linear.weight._grad_ivar().dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)

        self.linear.to()
        self.assertEqual(self.linear.weight.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        self.assertEqual(self.linear.buf_name.dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        np.testing.assert_allclose(self.linear.weight.grad.numpy(),
                                   self.new_grad,
                                   rtol=1e-05)
        self.assertEqual(self.linear.weight._grad_ivar().dtype,
                         paddle.fluid.core.VarDesc.VarType.FP64)
        for p in self.linear.parameters():
            if in_dygraph_mode():
                self.assertTrue(
                    isinstance(p, paddle.fluid.framework.EagerParamBase))
            else:
                self.assertTrue(isinstance(p, paddle.fluid.framework.ParamBase))

    def func_test_to_api_none_buffer(self):
        model = paddle.nn.Linear(2, 4)
        buffer = None
        model.register_buffer("buf_name", buffer, persistable=True)
        model.to(dtype='float64')
        self.assertEqual(model._buffers['buf_name'], None)

    def test_main(self):
        with _test_eager_guard():
            self.funcsetUp()
            self.func_test_to_api()
            self.func_test_to_api_paddle_dtype()
            self.func_test_to_api_numpy_dtype()
            self.func_test_to_api_none_buffer()
        self.funcsetUp()
        self.func_test_to_api()
        self.func_test_to_api_paddle_dtype()
        self.func_test_to_api_numpy_dtype()
        self.func_test_to_api_none_buffer()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
