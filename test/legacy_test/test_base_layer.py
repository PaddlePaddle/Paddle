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
import sys
import unittest

import numpy as np
from op_test import get_device, get_device_place, is_custom_device

import paddle
from paddle import base
from paddle.base.framework import EagerParamBase

sys.path.append("../dygraph_to_static")
from dygraph_to_static_utils import enable_to_static_guard


class L1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._param_attr = base.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.1)
        )
        self.w1 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False
        )
        self.w2 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False
        )

    def forward(self):
        return self.w1 + self.w2


class L2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = L1()
        self.layer2 = L1()

    def forward(self):
        return self.layer1() + self.layer2()


class L3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = L2()
        self.layer2 = L2()

    def forward(self):
        return self.layer1() + self.layer2()


class TestBaseLayer(unittest.TestCase):
    def test_one_level(self):
        l = L1()
        ret = l()
        expected_names = ['l1.w1', 'l1.w2']
        idx = 0
        for name, _ in l.named_parameters(prefix='l1'):
            self.assertEqual(name, expected_names[idx])
            idx += 1
        np.testing.assert_allclose(
            ret.numpy(), 0.2 * np.ones([2, 2]), rtol=1e-05
        )

    def test_three_level(self):
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
        np.testing.assert_allclose(
            ret.numpy(), 0.8 * np.ones([2, 2]), rtol=1e-05
        )

    def test_add_parameter_with_error(self):
        net = paddle.nn.Layer()
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


class BufferLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        buffer_var = paddle.to_tensor(np.zeros([2, 4]).astype('int32'))
        self.register_buffer("layer_buffer", buffer_var)

    def forward(self):
        pass


class BufferNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.buffer_layer = BufferLayer()
        self.w1 = self.create_parameter(
            shape=[2, 2], dtype='float32', is_bias=False
        )
        buffer_var = paddle.to_tensor(np.ones([2, 4]).astype('int32'))
        self.register_buffer("net_buffer", buffer_var)

        self.new_buffer = paddle.to_tensor(np.ones([4, 2]).astype('int32'))

    def forward(self):
        pass


class TestBuffer(unittest.TestCase):
    def test_buffers_and_named_buffers(self):
        def names(named_buffers):
            return [name for name, _ in named_buffers]

        layer = BufferLayer()
        net = BufferNet()

        self.assertEqual(len(layer.buffers()), 1)
        self.assertEqual(names(layer.named_buffers()), ['layer_buffer'])

        self.assertEqual(len(net.buffers()), 3)
        self.assertEqual(
            names(net.named_buffers()),
            ['net_buffer', 'new_buffer', 'buffer_layer.layer_buffer'],
        )

        self.assertEqual(len(net.buffers(include_sublayers=False)), 2)
        self.assertEqual(
            names(net.named_buffers(include_sublayers=False)),
            ['net_buffer', 'new_buffer'],
        )

    def test_register_buffer_with_error(self):
        net = paddle.nn.Layer()
        var = paddle.to_tensor(np.zeros([1]))

        with self.assertRaisesRegex(
            TypeError, "name of buffer should be a string"
        ):
            net.register_buffer(12, var)

        with self.assertRaisesRegex(
            TypeError, "buffer should be a Paddle.Tensor"
        ):
            net.register_buffer(
                "buffer_name", EagerParamBase([2, 2], 'float32')
            )

        with self.assertRaisesRegex(KeyError, "name of buffer can not contain"):
            net.register_buffer("buffer.name", var)

        with self.assertRaisesRegex(
            KeyError, "name of buffer can not be empty"
        ):
            net.register_buffer("", var)

        net.attr_name = 10
        with self.assertRaisesRegex(KeyError, "already exists"):
            net.register_buffer("attr_name", var)

        del net.attr_name
        net.attr_name = EagerParamBase([2, 2], 'float32')
        with self.assertRaisesRegex(KeyError, "already exists"):
            net.register_buffer("attr_name", var)

    def test_register_buffer_same_name(self):
        net = paddle.nn.Layer()
        var1 = paddle.to_tensor(np.zeros([1]))
        var2 = paddle.to_tensor(np.zeros([2]))
        var3 = paddle.to_tensor(np.zeros([3]))

        net.register_buffer("buffer_name", var1)
        self.assert_var_base_equal(net.buffer_name, var1)
        net.register_buffer("buffer_name", var2)
        self.assert_var_base_equal(net.buffer_name, var2)
        net.register_buffer("buffer_name", var3)
        self.assert_var_base_equal(net.buffer_name, var3)

    def test_buffer_not_persistable(self):
        net = paddle.nn.Layer()
        var1 = paddle.to_tensor(np.zeros([1]))

        net.register_buffer("buffer_name", var1, persistable=False)
        self.assertEqual(len(net.buffers()), 1)
        self.assertEqual(len(net.state_dict()), 0)

    def test_buffer_not_persistable_del(self):
        net = paddle.nn.Layer()
        var1 = paddle.to_tensor(np.zeros([1]))
        net.register_buffer("buffer_name", var1, persistable=False)
        del net.buffer_name
        self.assertEqual(len(net.buffers()), 0)

    def test_buffer_not_persistable_overwrite(self):
        net = paddle.nn.Layer()
        var1 = paddle.to_tensor(np.zeros([1]))
        var2 = paddle.to_tensor(np.zeros([2]))
        net.register_buffer("buffer_name", var1, persistable=False)
        net.register_buffer("buffer_name", var2)

        # Allow to overwrite a non-persistable buffer with a persistable var.
        self.assertEqual(len(net.buffers()), 1)
        self.assertEqual(len(net.state_dict()), 1)

        net.register_buffer("buffer_name", var1, persistable=False)
        self.assertEqual(len(net.buffers()), 1)
        self.assertEqual(len(net.state_dict()), 0)

    def test_buffer_not_persistable_assign(self):
        net = paddle.nn.Layer()
        var1 = paddle.to_tensor(np.zeros([1]))
        net.register_buffer("buffer_name", var1, persistable=False)

        # Assigning Nones will remove the buffer, but allow to re-assign
        # to remark it as buffer.
        net.buffer_name = None
        self.assertEqual(len(net.buffers()), 0)
        self.assertEqual(len(net.state_dict()), 0)

        net.buffer_name = var1
        self.assertEqual(len(net.buffers()), 1)
        self.assertEqual(len(net.state_dict()), 0)

        # Re-assign a EagerParamBase will remove the buffer.
        net.buffer_name = EagerParamBase([2, 2], 'float32')
        self.assertEqual(len(net.buffers()), 0)
        self.assertEqual(len(net.state_dict()), 1)

    def test_buffer_not_persistable_load(self):
        net = paddle.nn.Layer()
        var1 = paddle.to_tensor(np.zeros([1]))
        net.register_buffer("buffer_name", var1, persistable=False)
        net.load_dict({})

    def test_buffer_state_dict(self):
        net = paddle.nn.Layer()
        var1 = paddle.to_tensor(np.zeros([2, 3]))
        var2 = paddle.to_tensor(np.zeros([3, 2]))
        net.register_buffer("buffer_var1", var1)
        net.register_buffer("buffer_var2", var2, persistable=False)

        self.assertEqual(len(net.state_dict()), 1)
        self.assertEqual(
            [name for name, _ in net.state_dict().items()], ["buffer_var1"]
        )

        # load state_dict
        net_load = paddle.nn.Layer()
        var = paddle.to_tensor(np.ones([2, 3]))
        net_load.register_buffer("buffer_var1", var)
        net_load.load_dict(net.state_dict())

        self.assert_var_base_equal(net_load.buffer_var1, var1)

    def assert_var_base_equal(self, var1, var2):
        np.testing.assert_array_equal(var1.numpy(), var2.numpy())


class TestStateDictHook(unittest.TestCase):
    def test_state_dict_pre_hook(self):
        with base.dygraph.guard():
            layer = paddle.nn.Layer()
            parameter = layer.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            layer.register_parameter("weight", parameter)

            hook_calls = []

            def state_dict_pre_hook(layer, prefix, keep_vars):
                hook_calls.append((layer, prefix, keep_vars))

            hook_remove_helper = layer.register_state_dict_pre_hook(
                state_dict_pre_hook
            )
            state_dict = layer.state_dict(prefix="prefix.", keep_vars=False)
            self.assertIn("prefix.weight", state_dict)
            self.assertEqual(hook_calls, [(layer, "prefix.", False)])
            hook_remove_helper.remove()
            self.assertNotIn(
                hook_remove_helper._hook_id, layer._state_dict_pre_hooks
            )

    def test_state_dict_post_hook(self):
        with base.dygraph.guard():
            layer = paddle.nn.Layer()
            parameter = layer.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            layer.register_parameter("weight", parameter)

            hook_calls = []

            def state_dict_post_hook(destination):
                hook_calls.append(destination)
                destination["post_hook_weight"] = destination.pop("weight")

            hook_remove_helper = layer.register_state_dict_post_hook(
                state_dict_post_hook
            )
            state_dict = layer.state_dict()
            self.assertIn("post_hook_weight", state_dict)
            self.assertNotIn("weight", state_dict)
            self.assertEqual(hook_calls, [state_dict])
            hook_remove_helper.remove()
            self.assertNotIn(
                hook_remove_helper._hook_id, layer._state_dict_hooks
            )

    def test_state_dict_post_hook_with_torch_signature(self):
        with base.dygraph.guard():
            layer = paddle.nn.Layer()
            parameter = layer.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            layer.register_parameter("weight", parameter)

            child = paddle.nn.Layer()
            child_parameter = child.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            child.register_parameter("weight", child_parameter)
            layer.add_sublayer("child", child)

            hook_calls = []
            child_hook_calls = []

            def state_dict_post_hook(
                module, destination, prefix, local_metadata
            ):
                hook_calls.append((module, destination, prefix, local_metadata))
                destination[prefix + "post_hook_weight"] = destination.pop(
                    prefix + "weight"
                )

            def child_state_dict_post_hook(
                module, destination, prefix, local_metadata
            ):
                child_hook_calls.append(
                    (module, destination, prefix, local_metadata)
                )
                destination[prefix + "post_hook_weight"] = destination.pop(
                    prefix + "weight"
                )

            hook_remove_helper = layer.register_state_dict_post_hook(
                state_dict_post_hook
            )
            child_hook_remove_helper = child.register_state_dict_post_hook(
                child_state_dict_post_hook
            )

            state_dict = layer.state_dict(prefix="prefix.", keep_vars=False)

            self.assertIn("prefix.post_hook_weight", state_dict)
            self.assertIn("prefix.child.post_hook_weight", state_dict)
            self.assertNotIn("prefix.weight", state_dict)
            self.assertNotIn("prefix.child.weight", state_dict)
            self.assertEqual(hook_calls, [(layer, state_dict, "prefix.", {})])
            self.assertEqual(
                child_hook_calls,
                [(child, state_dict, "prefix.child.", {})],
            )

            hook_remove_helper.remove()
            child_hook_remove_helper.remove()
            self.assertNotIn(
                hook_remove_helper._hook_id, layer._state_dict_hooks
            )
            self.assertNotIn(
                child_hook_remove_helper._hook_id, child._state_dict_hooks
            )

    def test_state_dict_post_hook_return_with_torch_signature(self):
        with base.dygraph.guard():
            layer = paddle.nn.Layer()
            parameter = layer.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            layer.register_parameter("weight", parameter)

            def state_dict_post_hook(
                module, destination, prefix, local_metadata
            ):
                return {"replacement": destination[prefix + "weight"]}

            layer.register_state_dict_post_hook(state_dict_post_hook)
            state_dict = layer.state_dict()

            self.assertEqual(list(state_dict.keys()), ["replacement"])

    def test_load_state_dict_hooks(self):
        with base.dygraph.guard():
            layer = paddle.nn.Layer()
            parameter = layer.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            layer.register_parameter("weight", parameter)

            child = paddle.nn.Layer()
            child_parameter = child.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            child.register_parameter("weight", child_parameter)
            layer.add_sublayer("child", child)

            pre_hook_calls = []
            post_hook_calls = []
            child_pre_hook_calls = []
            child_post_hook_calls = []

            def load_state_dict_pre_hook(
                layer,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                pre_hook_calls.append((layer, prefix, local_metadata, strict))

            def child_load_state_dict_pre_hook(
                layer,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                child_pre_hook_calls.append(
                    (layer, prefix, local_metadata, strict)
                )

            def load_state_dict_post_hook(layer, incompatible_keys):
                post_hook_calls.append((layer, incompatible_keys.missing_keys))

            def child_load_state_dict_post_hook(layer, incompatible_keys):
                child_post_hook_calls.append(
                    (layer, incompatible_keys.missing_keys)
                )

            pre_hook = layer.register_load_state_dict_pre_hook(
                load_state_dict_pre_hook
            )
            post_hook = layer.register_load_state_dict_post_hook(
                load_state_dict_post_hook
            )
            child_pre_hook = child.register_load_state_dict_pre_hook(
                child_load_state_dict_pre_hook
            )
            child_post_hook = child.register_load_state_dict_post_hook(
                child_load_state_dict_post_hook
            )

            incompatible_keys = layer.load_state_dict(
                {
                    "weight": paddle.ones_like(parameter),
                    "child.weight": paddle.ones_like(child_parameter),
                },
                strict=True,
            )
            self.assertEqual(incompatible_keys.missing_keys, [])
            self.assertEqual(incompatible_keys.unexpected_keys, [])
            self.assertEqual(pre_hook_calls, [(layer, "", {}, True)])
            self.assertEqual(
                child_pre_hook_calls, [(child, "child.", {}, True)]
            )
            self.assertEqual(post_hook_calls, [(layer, [])])
            self.assertEqual(child_post_hook_calls, [(child, [])])

            pre_hook.remove()
            post_hook.remove()
            child_pre_hook.remove()
            child_post_hook.remove()
            self.assertNotIn(
                pre_hook._hook_id, layer._load_state_dict_pre_hooks
            )
            self.assertNotIn(
                post_hook._hook_id, layer._load_state_dict_post_hooks
            )
            self.assertNotIn(
                child_pre_hook._hook_id, child._load_state_dict_pre_hooks
            )
            self.assertNotIn(
                child_post_hook._hook_id, child._load_state_dict_post_hooks
            )

    def test_load_state_dict_post_hook_return(self):
        with base.dygraph.guard():
            layer = paddle.nn.Layer()
            parameter = layer.create_parameter(
                shape=[1], dtype='float32', is_bias=False
            )
            layer.register_parameter("weight", parameter)

            def load_state_dict_post_hook(layer, incompatible_keys):
                return incompatible_keys

            layer.register_load_state_dict_post_hook(load_state_dict_post_hook)
            with self.assertRaisesRegex(
                AssertionError,
                "Hooks registered with ``register_load_state_dict_post_hook``",
            ):
                layer.load_state_dict(
                    {"weight": paddle.ones_like(parameter)}, strict=True
                )

    def test_extra_state_state_dict(self):
        class ExtraStateLayer(paddle.nn.Layer):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def get_extra_state(self):
                return {"value": self.value}

            def set_extra_state(self, state):
                self.value = state["value"]

        with base.dygraph.guard():
            layer = ExtraStateLayer(1)
            layer.child = ExtraStateLayer(2)

            state_dict = layer.state_dict(prefix="prefix.")

            self.assertEqual(state_dict["prefix._extra_state"], {"value": 1})
            self.assertEqual(
                state_dict["prefix.child._extra_state"], {"value": 2}
            )

    def test_extra_state_load_state_dict(self):
        class ExtraStateLayer(paddle.nn.Layer):
            def __init__(self, value):
                super().__init__()
                self.value = value
                parameter = self.create_parameter(
                    shape=[1], dtype='float32', is_bias=False
                )
                self.register_parameter("weight", parameter)

            def get_extra_state(self):
                return {"value": self.value}

            def set_extra_state(self, state):
                self.value = state["value"]

        with base.dygraph.guard():
            layer = ExtraStateLayer(0)
            layer.child = ExtraStateLayer(0)

            incompatible_keys = layer.load_state_dict(
                {
                    "weight": paddle.ones_like(layer.weight),
                    "_extra_state": {"value": 3},
                    "child.weight": paddle.ones_like(layer.child.weight),
                    "child._extra_state": {"value": 4},
                },
                strict=True,
            )

            self.assertEqual(layer.value, 3)
            self.assertEqual(layer.child.value, 4)
            np.testing.assert_array_equal(layer.weight.numpy(), np.ones([1]))
            np.testing.assert_array_equal(
                layer.child.weight.numpy(), np.ones([1])
            )
            self.assertEqual(incompatible_keys.missing_keys, [])
            self.assertEqual(incompatible_keys.unexpected_keys, [])

    def test_extra_state_strict_error(self):
        class ExtraStateLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.value = 0

            def get_extra_state(self):
                return {"value": self.value}

            def set_extra_state(self, state):
                self.value = state["value"]

        with base.dygraph.guard():
            layer = ExtraStateLayer()
            layer.child = ExtraStateLayer()

            with self.assertRaisesRegex(RuntimeError, "child._extra_state"):
                layer.load_state_dict(
                    {"_extra_state": {"value": 1}}, strict=True
                )

            layer = paddle.nn.Layer()
            with self.assertRaisesRegex(RuntimeError, "_extra_state"):
                layer.load_state_dict(
                    {"_extra_state": {"value": 1}}, strict=True
                )

    def test_extra_state_non_dict(self):
        class ExtraStateLayer(paddle.nn.Layer):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def get_extra_state(self):
                return self.value

            def set_extra_state(self, state):
                self.value = state

        with base.dygraph.guard():
            for state in ("value", 1, ExtraStateLayer(2)):
                layer = ExtraStateLayer(state)
                layer_load = ExtraStateLayer(None)

                layer_load.load_state_dict(layer.state_dict())

                self.assertEqual(layer_load.value, state)

    def test_none_extra_state_is_not_saved(self):
        class NoneExtraStateLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                parameter = self.create_parameter(
                    shape=[1], dtype='float32', is_bias=False
                )
                self.register_parameter("weight", parameter)

            def get_extra_state(self):
                return None

            def set_extra_state(self, state):
                self.value = state

        with base.dygraph.guard():
            state_dict = NoneExtraStateLayer().state_dict()

            self.assertIn("weight", state_dict)
            self.assertNotIn("_extra_state", state_dict)
            for value in state_dict.values():
                self.assertIsNotNone(value)

    def test_extra_state_missing_method(self):
        class MissingSetExtraStateLayer(paddle.nn.Layer):
            def get_extra_state(self):
                return {"value": 1}

        class MissingGetExtraStateLayer(paddle.nn.Layer):
            def set_extra_state(self, state):
                self.value = state["value"]

        with base.dygraph.guard():
            layer = MissingSetExtraStateLayer()
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                layer.load_state_dict(layer.state_dict())

            layer = MissingGetExtraStateLayer()
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                layer.load_state_dict(layer.state_dict())

    def test_extra_state_with_amp_state_dict_hook(self):
        class ExtraStateLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                parameter = self.create_parameter(
                    shape=[1], dtype='float32', is_bias=False
                )
                self.register_parameter("weight", parameter)

            def get_extra_state(self):
                return {"value": 1}

            def set_extra_state(self, state):
                self.value = state["value"]

        with base.dygraph.guard():
            layer = ExtraStateLayer()
            layer = paddle.amp.decorate(
                models=layer, level='O2', save_dtype='float64'
            )

            state_dict = layer.state_dict()

            self.assertEqual(state_dict["_extra_state"], {"value": 1})

    def test_default_extra_state(self):
        with base.dygraph.guard():
            layer = paddle.nn.Layer()

            self.assertNotIn("_extra_state", layer.state_dict())
            with self.assertRaisesRegex(RuntimeError, "get_extra_state"):
                layer.get_extra_state()
            with self.assertRaisesRegex(RuntimeError, "set_extra_state"):
                layer.set_extra_state(None)


class BufferNetWithModification(paddle.nn.Layer):
    def __init__(self, shape):
        super().__init__()

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
        self.shape = [10, 16]

    def _run(self, to_static=False):
        with enable_to_static_guard(to_static):
            x = paddle.ones([1], 'int32')
            net = BufferNetWithModification(self.shape)
            out = net(x)

            return out, net.buffer1, net.buffer2

    def test_modified(self):
        self.funcsetUp()
        dy_outs = self._run(False)
        st_outs = self._run(True)

        for i in range(len(dy_outs)):
            np.testing.assert_array_equal(
                dy_outs[i].numpy(), st_outs[i].numpy()
            )


class TestLayerTo(unittest.TestCase):
    def funcsetUp(self):
        self.linear = paddle.nn.Linear(2, 2)
        self.new_grad = np.random.random([2, 2])
        self.linear.weight._set_grad_ivar(paddle.to_tensor(self.new_grad))
        buffer = paddle.to_tensor([0.0], dtype='float32')
        self.linear.register_buffer("buf_name", buffer, persistable=True)

        sublayer = paddle.nn.Conv1D(3, 2, 3)
        self.linear.add_sublayer("1", sublayer)

    def func_test_to_api(self):
        if paddle.framework.use_pir_api():
            dtype_float64 = paddle.base.core.DataType.FLOAT64
        else:
            dtype_float64 = paddle.base.core.VarDesc.VarType.FP64
        self.linear.to(dtype='double')
        self.assertEqual(self.linear.weight.dtype, dtype_float64)
        self.assertEqual(self.linear.buf_name.dtype, dtype_float64)
        np.testing.assert_allclose(
            self.linear.weight.grad.numpy(), self.new_grad, rtol=1e-05
        )
        self.assertEqual(
            self.linear.weight._grad_ivar().dtype,
            dtype_float64,
        )

        self.linear.to()
        self.assertEqual(self.linear.weight.dtype, dtype_float64)
        self.assertEqual(self.linear.buf_name.dtype, dtype_float64)
        np.testing.assert_allclose(
            self.linear.weight.grad.numpy(), self.new_grad, rtol=1e-05
        )
        self.assertEqual(
            self.linear.weight._grad_ivar().dtype,
            dtype_float64,
        )
        for p in self.linear.parameters():
            self.assertTrue(isinstance(p, paddle.base.framework.EagerParamBase))

        if paddle.base.is_compiled_with_cuda():
            self.linear.to(device=get_device_place())
            self.assertTrue(self.linear.weight.place.is_gpu_place())
            self.assertEqual(self.linear.weight.place.gpu_device_id(), 0)
            self.assertTrue(self.linear.buf_name.place.is_gpu_place())
            self.assertEqual(self.linear.buf_name.place.gpu_device_id(), 0)
            self.assertTrue(
                self.linear.weight._grad_ivar().place.is_gpu_place()
            )
            self.assertEqual(
                self.linear.weight._grad_ivar().place.gpu_device_id(), 0
            )

            self.linear.to(device=get_device(True))
            self.assertTrue(self.linear.weight.place.is_gpu_place())
            self.assertEqual(self.linear.weight.place.gpu_device_id(), 0)
            self.assertTrue(self.linear.buf_name.place.is_gpu_place())
            self.assertEqual(self.linear.buf_name.place.gpu_device_id(), 0)
            self.assertTrue(
                self.linear.weight._grad_ivar().place.is_gpu_place()
            )
            self.assertEqual(
                self.linear.weight._grad_ivar().place.gpu_device_id(), 0
            )
            for p in self.linear.parameters():
                self.assertTrue(
                    isinstance(p, paddle.base.framework.EagerParamBase)
                )
        elif is_custom_device():
            self.linear.to(device=get_device_place())
            self.assertTrue(self.linear.weight.place.is_custom_place())
            self.assertEqual(self.linear.weight.place.custom_device_id(), 0)
            self.assertTrue(self.linear.buf_name.place.is_custom_place())
            self.assertEqual(self.linear.buf_name.place.custom_device_id(), 0)
            self.assertTrue(
                self.linear.weight._grad_ivar().place.is_custom_place()
            )
            self.assertEqual(
                self.linear.weight._grad_ivar().place.custom_device_id(), 0
            )

            self.linear.to(device=get_device(True))
            self.assertTrue(self.linear.weight.place.is_custom_place())
            self.assertEqual(self.linear.weight.place.custom_device_id(), 0)
            self.assertTrue(self.linear.buf_name.place.is_custom_place())
            self.assertEqual(self.linear.buf_name.place.custom_device_id(), 0)
            self.assertTrue(
                self.linear.weight._grad_ivar().place.is_custom_place()
            )
            self.assertEqual(
                self.linear.weight._grad_ivar().place.custom_device_id(), 0
            )
            for p in self.linear.parameters():
                self.assertTrue(
                    isinstance(p, paddle.base.framework.EagerParamBase)
                )

        self.linear.to(device=paddle.CPUPlace())
        self.assertTrue(self.linear.weight.place.is_cpu_place())
        self.assertTrue(self.linear.buf_name.place.is_cpu_place())
        self.assertTrue(self.linear.weight._grad_ivar().place.is_cpu_place())

        self.linear.to(device='cpu')
        self.assertTrue(self.linear.weight.place.is_cpu_place())
        self.assertTrue(self.linear.buf_name.place.is_cpu_place())
        self.assertTrue(self.linear.weight._grad_ivar().place.is_cpu_place())

        self.assertRaises(ValueError, self.linear.to, device=1)

        self.assertRaises(TypeError, self.linear.to, blocking=1)
        self.assertRaises(TypeError, self.linear.to, non_blocking=0)

    def func_test_to_api_paddle_dtype(self):
        if paddle.framework.use_pir_api():
            dtype_float64 = paddle.base.core.DataType.FLOAT64
        else:
            dtype_float64 = paddle.base.core.VarDesc.VarType.FP64

        self.linear.to(dtype=paddle.float64)
        self.assertEqual(self.linear.weight.dtype, dtype_float64)
        self.assertEqual(self.linear.buf_name.dtype, dtype_float64)
        np.testing.assert_allclose(
            self.linear.weight.grad.numpy(), self.new_grad, rtol=1e-05
        )
        self.assertEqual(
            self.linear.weight._grad_ivar().dtype,
            dtype_float64,
        )

        self.linear.to()
        self.assertEqual(self.linear.weight.dtype, dtype_float64)
        self.assertEqual(self.linear.buf_name.dtype, dtype_float64)
        np.testing.assert_allclose(
            self.linear.weight.grad.numpy(), self.new_grad, rtol=1e-05
        )
        self.assertEqual(
            self.linear.weight._grad_ivar().dtype,
            dtype_float64,
        )
        for p in self.linear.parameters():
            self.assertTrue(isinstance(p, paddle.base.framework.EagerParamBase))

    def func_test_to_api_numpy_dtype(self):
        if paddle.framework.use_pir_api():
            dtype_float64 = paddle.base.core.DataType.FLOAT64
        else:
            dtype_float64 = paddle.base.core.VarDesc.VarType.FP64

        self.linear.to(dtype=np.float64)
        self.assertEqual(self.linear.weight.dtype, dtype_float64)
        self.assertEqual(self.linear.buf_name.dtype, dtype_float64)
        np.testing.assert_allclose(
            self.linear.weight.grad.numpy(), self.new_grad, rtol=1e-05
        )
        self.assertEqual(
            self.linear.weight._grad_ivar().dtype,
            dtype_float64,
        )

        self.linear.to()
        self.assertEqual(self.linear.weight.dtype, dtype_float64)
        self.assertEqual(self.linear.buf_name.dtype, dtype_float64)
        np.testing.assert_allclose(
            self.linear.weight.grad.numpy(), self.new_grad, rtol=1e-05
        )
        self.assertEqual(
            self.linear.weight._grad_ivar().dtype,
            dtype_float64,
        )
        for p in self.linear.parameters():
            self.assertTrue(isinstance(p, paddle.base.framework.EagerParamBase))

    def func_test_to_api_none_buffer(self):
        model = paddle.nn.Linear(2, 4)
        buffer = None
        model.register_buffer("buf_name", buffer, persistable=True)
        model.to(dtype='float64')
        self.assertIsNone(model._buffers['buf_name'])

    def test_main(self):
        self.funcsetUp()
        self.func_test_to_api()
        self.func_test_to_api_paddle_dtype()
        self.func_test_to_api_numpy_dtype()
        self.func_test_to_api_none_buffer()


if __name__ == '__main__':
    unittest.main()
