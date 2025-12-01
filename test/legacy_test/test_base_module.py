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
from collections import OrderedDict

import paddle
from paddle import Tensor, nn


class Module(nn.Module):
    # class Module(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SubModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(10, 5)


class SubModule1(nn.Module):
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()
        self.register_buffer('running_mean', paddle.zeros([output_size]))

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class NestedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SubModule1(10, 8)
        self.layer2 = SubModule1(8, 6)
        self.layer3 = SubModule1(6, 4)
        self.final_layer = nn.Linear(4, 2)


class TestRegisterBuffer(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_register_buffer_basic(self):
        buffer_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        self.module.register_buffer('test_buffer', buffer_tensor)
        self.assertTrue(hasattr(self.module, 'test_buffer'))
        self.assertIn('test_buffer', self.module._buffers)

        all_buffers = list(self.module.buffers(recurse=True))
        self.assertTrue(len(all_buffers) > 0)
        self.assertTrue(any(isinstance(b, paddle.Tensor) for b in all_buffers))

    def test_register_buffer_persistent(self):
        persistent_buffer = paddle.to_tensor([4.0, 5.0, 6.0])
        self.module.register_buffer(
            'persistent_buf', persistent_buffer, persistent=True
        )
        self.assertIn('persistent_buf', self.module._buffers)
        self.assertNotIn(
            'persistent_buf', self.module._non_persistent_buffers_set
        )

    def test_register_buffer_non_persistent(self):
        non_persistent_buffer = paddle.to_tensor([7.0, 8.0, 9.0])
        self.module.register_buffer(
            'non_persistent_buf', non_persistent_buffer, persistent=False
        )
        self.assertIn('non_persistent_buf', self.module._buffers)
        self.assertIn(
            'non_persistent_buf', self.module._non_persistent_buffers_set
        )


class TestRegisterParameter(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_register_parameter_basic(self):
        param_tensor = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        param = nn.Parameter(param_tensor)

        self.module.register_parameter('test_param', param)

        self.assertTrue(hasattr(self.module, 'test_param'))
        self.assertIn('test_param', self.module._parameters)
        self.assertTrue(paddle.allclose(self.module.test_param, param))
        self.assertTrue(self.module.test_param.trainable)

    def test_register_parameter_none(self):
        self.module.register_parameter('none_param', None)

        self.assertTrue(hasattr(self.module, 'none_param'))
        self.assertIn('none_param', self.module._parameters)
        self.assertIsNone(self.module.none_param)

    def test_register_parameter_with_tensor(self):
        param_tensor = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)

        self.module.register_parameter('test_param', nn.Parameter(param_tensor))

        self.assertIn('test_param', self.module._parameters)
        self.assertIsInstance(self.module.test_param, nn.Parameter)
        self.assertTrue(paddle.allclose(self.module.test_param, param_tensor))


class TestAddModule(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_add_module_basic(self):
        submodule = nn.Linear(10, 5)
        self.module.add_module('linear', submodule)

        self.assertTrue(hasattr(self.module, 'linear'))
        self.assertIn('linear', self.module._modules)
        self.assertEqual(self.module.linear, submodule)
        self.assertIsInstance(self.module.linear, nn.Linear)

    def test_register_module_basic(self):
        submodule = nn.Linear(10, 5)
        self.module.register_module('linear', submodule)

        self.assertTrue(hasattr(self.module, 'linear'))
        self.assertIn('linear', self.module._modules)
        self.assertEqual(self.module.linear, submodule)
        self.assertIsInstance(self.module.linear, nn.Linear)

    def test_add_module_none(self):
        self.module.add_module('empty_module', None)

        self.assertTrue(hasattr(self.module, 'empty_module'))
        self.assertIn('empty_module', self.module._modules)
        self.assertIsNone(self.module.empty_module)

    def test_add_module_hierarchy(self):
        child_module = SubModule()
        self.module.add_module('child', child_module)

        self.assertIn('child', self.module._modules)
        self.assertIn('linear', child_module._modules)
        self.assertIsInstance(child_module.linear, nn.Linear)

    def test_module_forward(self):
        submodule = nn.Linear(10, 5)
        self.module.add_module('linear', submodule)

        input_tensor = paddle.ones([2, 10])
        output = self.module.linear(input_tensor)

        self.assertEqual(output.shape, [2, 5])
        self.assertIsInstance(output, Tensor)


class TestGetSubmodule(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_get_submodule_basic(self):
        submodule = nn.Linear(10, 5)
        self.module.add_module('linear', submodule)

        retrieved_module = self.module.get_submodule('linear')
        self.assertEqual(retrieved_module, submodule)
        self.assertIsInstance(retrieved_module, nn.Linear)

    def test_get_submodule_nested(self):
        nested_module = NestedModule()
        self.module.add_module('nested', nested_module)

        test_cases = [
            'nested',
            'nested.layer1',
            'nested.layer1.linear',
            'nested.layer2.activation',
            'nested.final_layer',
        ]

        for target in test_cases:
            with self.subTest(target=target):
                module = self.module.get_submodule(target)
                self.assertIsInstance(module, nn.Module)

    def test_get_submodule_empty_target(self):
        result = self.module.get_submodule('')
        self.assertEqual(result, self.module)

    def test_get_parameter_basic(self):
        param_tensor = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        param = nn.Parameter(param_tensor)
        self.module.register_parameter('test_param', param)

        retrieved_param = self.module.get_parameter('test_param')
        self.assertIs(retrieved_param, param)
        self.assertIsInstance(retrieved_param, nn.Parameter)
        self.assertTrue(paddle.allclose(retrieved_param, param_tensor))

    def test_get_parameter_nested(self):
        nested_module = NestedModule()
        self.module.add_module('nested', nested_module)

        test_cases = [
            'nested.layer1.linear.weight',
            'nested.layer1.linear.bias',
            'nested.final_layer.weight',
            'nested.final_layer.bias',
        ]

        for target in test_cases:
            with self.subTest(target=target):
                param = self.module.get_parameter(target)
                self.assertIsInstance(param, nn.Parameter)
                self.assertTrue(param.trainable)

    def test_get_parameter_vs_get_submodule(self):
        nested_module = NestedModule()
        self.module.add_module('nested', nested_module)

        module = self.module.get_submodule('nested.layer1.linear')
        self.assertIsInstance(module, nn.Linear)

        weight_param = self.module.get_parameter('nested.layer1.linear.weight')
        self.assertIsInstance(weight_param, nn.Parameter)
        self.assertTrue(paddle.allclose(weight_param, module.weight))

    def test_get_parameter_gradients(self):
        nested_module = NestedModule()
        self.module.add_module('nested', nested_module)

        param = self.module.get_parameter('nested.layer1.linear.weight')

        x = paddle.ones([1, 10])
        y = nested_module.layer1.linear(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(param.grad)
        self.assertEqual(param.grad.shape, param.shape)

    def test_get_submodule_error(self):
        with self.assertRaises(AttributeError):
            self.module.get_submodule('invalid_name')

    def test_get_parameter_reeor(self):
        with self.assertRaises(AttributeError) as cm:
            self.module.get_parameter("nonexistent_param")
        self.assertIn("has no attribute `nonexistent_param`", str(cm.exception))

        self.module.fake_attr = "I am not a parameter"
        with self.assertRaises(AttributeError) as cm:
            self.module.get_parameter("fake_attr")
        self.assertIn("`fake_attr` is not an nn.Parameter", str(cm.exception))

    def test_get_extra_state_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            self.module.get_extra_state()


class TestSetSubmodule(unittest.TestCase):
    def setUp(self):
        self.module = NestedModule()

    def test_replace_top_level(self):
        new_module = SubModule1(8, 6)
        self.module.set_submodule("layer2", new_module)
        self.assertIs(self.module.layer2, new_module)

    def test_replace_nested_submodule(self):
        new_linear = nn.Linear(6, 4)
        self.module.set_submodule("layer3.linear", new_linear)
        self.assertIs(self.module.layer3.linear, new_linear)

    def test_add_new_non_strict(self):
        new_mod = nn.Linear(10, 10)
        self.module.set_submodule("extra", new_mod)
        self.assertIs(self.module.extra, new_mod)

    def test_strict_missing_attr_raises(self):
        new_mod = nn.Linear(1, 1)
        with self.assertRaises(AttributeError):
            self.module.set_submodule("not_exist", new_mod, strict=True)

    def test_non_module_input_raises(self):
        with self.assertRaises(ValueError):
            self.module.set_submodule("layer1", "not_a_module")

    def test_empty_target_raises(self):
        with self.assertRaises(ValueError):
            self.module.set_submodule("", nn.Linear(1, 1))

    def test_non_module_attr_raises(self):
        self.module.some_value = 10
        with self.assertRaises(AttributeError):
            self.module.set_submodule("some_value", nn.Linear(1, 1))


class TestLoadStateDict(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_load_state_dict_basic(self):
        self.module.register_parameter(
            'custom_param', nn.Parameter(paddle.ones([3, 3]))
        )
        self.module.register_buffer('custom_buffer', paddle.zeros([2, 2]))
        original_state = self.module.state_dict()

        with paddle.no_grad():
            self.module.custom_param.set_value(
                self.module.custom_param * 2 + 1.0
            )
            self.module.custom_buffer.set_value(paddle.ones([2, 2]))

        result = self.module.load_state_dict(original_state)

        current_state = self.module.state_dict()
        for key in original_state:
            self.assertTrue(
                paddle.allclose(original_state[key], current_state[key])
            )

        self.assertEqual(len(result.missing_keys), 0)
        self.assertEqual(len(result.unexpected_keys), 0)

    def test_load_state_dict_strict_mode(self):
        self.module.register_parameter(
            'test_param', nn.Parameter(paddle.ones([3]))
        )
        original_state = self.module.state_dict()

        modified_state = original_state.copy()
        modified_state['extra_param'] = paddle.ones([5])
        modified_state.pop('test_param')

        with self.assertRaises(RuntimeError) as context:
            self.module.load_state_dict(modified_state, strict=True)

        error_msg = str(context.exception)
        self.assertIn("Missing key(s)", error_msg)
        self.assertIn("Unexpected key(s)", error_msg)

    def test_load_state_dict_non_strict_mode(self):
        self.module.register_parameter('param1', nn.Parameter(paddle.ones([3])))
        self.module.register_buffer('buffer1', paddle.zeros([2]))
        original_state = self.module.state_dict()

        modified_state = original_state.copy()
        modified_state['extra_param'] = paddle.ones([5])
        modified_state.pop('buffer1')

        result = self.module.load_state_dict(modified_state, strict=False)

        self.assertIn('buffer1', result.missing_keys)
        self.assertIn('extra_param', result.unexpected_keys)

        self.assertTrue(paddle.allclose(self.module.param1, paddle.ones([3])))


class TestNamedParameters(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_named_parameters_basic(self):
        param1 = paddle.create_parameter([3, 4], dtype='float32')
        param2 = paddle.create_parameter([2, 2], dtype='float32')
        self.module.register_parameter('weight', param1)
        self.module.register_parameter('bias', param2)

        named_params = dict(self.module.named_parameters())

        self.assertIn('weight', named_params)
        self.assertIn('bias', named_params)
        self.assertEqual(len(named_params), 2)

        self.assertTrue(paddle.allclose(named_params['weight'], param1))
        self.assertTrue(paddle.allclose(named_params['bias'], param2))

    def test_named_parameters_with_prefix(self):
        param = paddle.create_parameter([5], dtype='float32')
        self.module.register_parameter('test_param', param)

        names_without_prefix = [
            name for name, _ in self.module.named_parameters(prefix="")
        ]
        self.assertEqual(names_without_prefix, ['test_param'])

        names_with_prefix = [
            name for name, _ in self.module.named_parameters(prefix="module")
        ]
        self.assertEqual(names_with_prefix, ['module.test_param'])

    def test_named_parameters_recurse_false(self):
        sublayer = nn.Linear(10, 5)
        self.module.add_sublayer('linear', sublayer)

        main_param = paddle.create_parameter([3], dtype='float32')
        self.module.register_parameter('main_param', main_param)

        non_recurse_names = [
            name for name, _ in self.module.named_parameters(recurse=False)
        ]
        self.assertEqual(non_recurse_names, ['main_param'])

        recurse_names = [
            name for name, _ in self.module.named_parameters(recurse=True)
        ]
        self.assertIn('main_param', recurse_names)
        self.assertIn('linear.weight', recurse_names)
        self.assertIn('linear.bias', recurse_names)

    def test_named_parameters_remove_duplicate(self):
        shared_param = paddle.create_parameter([4, 4], dtype='float32')

        self.module.register_parameter('weight1', shared_param)
        self.module.register_parameter('weight2', shared_param)

        without_duplicate = list(
            self.module.named_parameters(remove_duplicate=False)
        )
        names_no_dedup = [name for name, _ in without_duplicate]
        self.assertEqual(len(names_no_dedup), 2)
        self.assertIn('weight1', names_no_dedup)
        self.assertIn('weight2', names_no_dedup)

        param_dict = dict(without_duplicate)
        self.assertIs(param_dict['weight1'], param_dict['weight2'])

        with_duplicate = list(
            self.module.named_parameters(remove_duplicate=True)
        )
        names_dedup = [name for name, _ in with_duplicate]
        self.assertEqual(len(names_dedup), 1)
        self.assertIn('weight1', names_dedup)

    def test_named_parameters_empty_module(self):
        named_params = list(self.module.named_parameters())
        self.assertEqual(len(named_params), 0)

    def test_named_parameters_complex_hierarchy(self):
        child1 = nn.Linear(10, 8)
        child2 = nn.Linear(8, 6)
        child1.add_sublayer('child2', child2)
        self.module.add_sublayer('child1', child1)

        self.module.register_parameter(
            'global_param', paddle.create_parameter([3], dtype='float32')
        )

        all_names = [name for name, _ in self.module.named_parameters()]

        expected_names = [
            'global_param',
            'child1.weight',
            'child1.bias',
            'child1.child2.weight',
            'child1.child2.bias',
        ]

        self.assertEqual(set(all_names), set(expected_names))
        self.assertEqual(len(all_names), len(expected_names))

    def test_named_parameters_with_buffers(self):
        param = paddle.create_parameter([5], dtype='float32')
        buffer = paddle.to_tensor([1, 2, 3])

        self.module.register_parameter('param', param)
        self.module.register_buffer('buffer', buffer)

        param_names = [name for name, _ in self.module.named_parameters()]
        buffer_names = [name for name, _ in self.module.named_buffers()]

        self.assertEqual(param_names, ['param'])
        self.assertEqual(buffer_names, ['buffer'])

        self.assertNotIn('buffer', param_names)
        self.assertNotIn('param', buffer_names)


class TestNamedModules(unittest.TestCase):
    def setUp(self):
        self.module = Module()

    def test_modules_basic(self):
        child1 = SubModule()
        child2 = nn.ReLU()
        self.module.add_sublayer('submodule', child1)
        self.module.add_sublayer('activation', child2)

        modules = list(self.module.modules())

        self.assertEqual(len(modules), 4)

    def test_named_modules_basic(self):
        child1 = SubModule()
        child2 = nn.ReLU()
        self.module.add_sublayer('submodule', child1)
        self.module.add_sublayer('activation', child2)

        named_modules = dict(self.module.named_modules())

        self.assertIn('', named_modules)
        self.assertIn('submodule', named_modules)
        self.assertIn('activation', named_modules)
        self.assertIn('submodule.linear', named_modules)
        self.assertEqual(len(named_modules), 4)

        self.assertIs(named_modules[''], self.module)
        self.assertIs(named_modules['submodule'], child1)
        self.assertIs(named_modules['activation'], child2)
        self.assertIs(named_modules['submodule.linear'], child1.linear)

    def test_named_modules_with_prefix(self):
        child = SubModule()
        self.module.add_sublayer('child', child)

        names_without_prefix = [
            name for name, _ in self.module.named_modules(prefix="")
        ]
        self.assertIn('', names_without_prefix)
        self.assertIn('child', names_without_prefix)
        self.assertIn('child.linear', names_without_prefix)

        names_with_prefix = [
            name for name, _ in self.module.named_modules(prefix="model")
        ]
        self.assertIn('model', names_with_prefix)
        self.assertIn('model.child', names_with_prefix)
        self.assertIn('model.child.linear', names_with_prefix)
        self.assertNotIn('', names_with_prefix)

    def test_named_modules_memo_parameter(self):
        nested_module = NestedModule()
        self.module.add_sublayer('nested', nested_module)

        memo_set = set()
        first_pass = list(self.module.named_modules(memo=memo_set))

        self.assertIn(self.module, memo_set)
        self.assertIn(nested_module, memo_set)
        self.assertIn(nested_module.layer1, memo_set)

        second_pass = list(self.module.named_modules(memo=memo_set))

        self.assertEqual(len(second_pass), 0)

    def test_named_modules_remove_duplicate(self):
        shared_module = nn.Dropout(0.5)

        self.module.add_sublayer('dropout1', shared_module)
        self.module.add_sublayer('dropout2', shared_module)

        without_dedup = list(self.module.named_modules(remove_duplicate=False))
        names_no_dedup = [name for name, _ in without_dedup]
        self.assertEqual(len(names_no_dedup), 3)
        self.assertIn('dropout1', names_no_dedup)
        self.assertIn('dropout2', names_no_dedup)

        module_dict = dict(without_dedup)
        self.assertIs(module_dict['dropout1'], module_dict['dropout2'])

        with_dedup = list(self.module.named_modules(remove_duplicate=True))
        names_dedup = [name for name, _ in with_dedup]
        self.assertEqual(len(names_dedup), 2)

        dropout_names = [name for name in names_dedup if 'dropout' in name]
        self.assertEqual(len(dropout_names), 1)

    def test_named_modules_complex_hierarchy(self):
        nested_module = NestedModule()
        self.module.add_sublayer('complex', nested_module)

        all_modules = dict(self.module.named_modules())

        expected_paths = {
            '',
            'complex',
            'complex.layer1',
            'complex.layer2',
            'complex.layer3',
            'complex.final_layer',
            'complex.layer1.linear',
            'complex.layer1.activation',
            'complex.layer2.linear',
            'complex.layer2.activation',
            'complex.layer3.linear',
            'complex.layer3.activation',
        }

        self.assertEqual(set(all_modules.keys()), expected_paths)

        self.assertIs(all_modules[''], self.module)
        self.assertIs(all_modules['complex'], nested_module)
        self.assertIs(all_modules['complex.layer1'], nested_module.layer1)
        self.assertIs(
            all_modules['complex.layer1.linear'], nested_module.layer1.linear
        )
        self.assertIs(
            all_modules['complex.final_layer'], nested_module.final_layer
        )

    def test_named_modules_empty_module(self):
        named_modules = list(self.module.named_modules())
        self.assertEqual(len(named_modules), 1)
        self.assertEqual(named_modules[0][0], '')
        self.assertIs(named_modules[0][1], self.module)


class TestGetBuffer(unittest.TestCase):
    def setUp(self):
        self.model = NestedModule()

    def test_get_existing_buffer(self):
        buf = self.model.get_buffer("layer1.running_mean")
        self.assertIsInstance(buf, paddle.Tensor)
        self.assertTrue(paddle.allclose(buf, paddle.zeros([8])))

    def test_get_nested_buffer(self):
        buf = self.model.get_buffer("layer2.running_mean")
        self.assertEqual(buf.shape[0], 6)

    def test_get_final_layer_error(self):
        with self.assertRaises(AttributeError):
            _ = self.model.get_buffer("final_layer.bias")

    def test_nonexistent_layer_error(self):
        with self.assertRaises(AttributeError):
            _ = self.model.get_buffer("nonexistent.running_mean")

    def test_nonexistent_buffer_error(self):
        with self.assertRaises(AttributeError):
            _ = self.model.get_buffer("layer1.not_a_buffer")


class TestModuleDeviceTransfer(unittest.TestCase):
    def setUp(self):
        self.model = NestedModule()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "Paddle not compiled with CUDA"
    )
    def test_cuda(self):
        model_gpu = self.model.cuda()
        for name, p in model_gpu.named_parameters():
            self.assertTrue(
                p.place.is_gpu_place(), f"{name} not on GPU with device=None"
            )
        for name, b in model_gpu.named_buffers():
            self.assertTrue(
                b.place.is_gpu_place(),
                f"{name} buffer not on GPU with device=None",
            )

        model_gpu = self.model.cuda(0)
        for name, p in model_gpu.named_parameters():
            self.assertTrue(
                p.place.is_gpu_place(), f"{name} not on GPU when using index 0"
            )
        for name, b in model_gpu.named_buffers():
            self.assertTrue(
                b.place.is_gpu_place(), f"{name} buffer not on GPU with index 0"
            )

        cuda_place = paddle.CUDAPlace(0)
        model_gpu = self.model.cuda(cuda_place)
        for name, p in model_gpu.named_parameters():
            self.assertTrue(
                p.place.is_gpu_place(),
                f"{name} not on GPU when using CUDAPlace",
            )
        for name, b in model_gpu.named_buffers():
            self.assertTrue(
                b.place.is_gpu_place(),
                f"{name} buffer not on GPU when using CUDAPlace",
            )

        with self.assertRaises(TypeError):
            self.model.cuda("gpu:0")

    @unittest.skipIf(
        not hasattr(paddle, "is_compiled_with_xpu")
        or not paddle.is_compiled_with_xpu(),
        "Paddle not built with XPU",
    )
    def test_xpu(self):
        model_xpu = self.model.xpu()
        for name, p in model_xpu.named_parameters():
            self.assertTrue(
                p.place.is_xpu_place(), f"{name} not moved to XPU (device=None)"
            )
        for name, b in model_xpu.named_buffers():
            self.assertTrue(
                b.place.is_xpu_place(),
                f"{name} buffer not moved to XPU (device=None)",
            )

        model_xpu = self.model.xpu(0)
        for name, p in model_xpu.named_parameters():
            self.assertTrue(
                p.place.is_xpu_place(), f"{name} not moved to XPU (device=0)"
            )
        for name, b in model_xpu.named_buffers():
            self.assertTrue(
                b.place.is_xpu_place(),
                f"{name} buffer not moved to XPU (device=0)",
            )

        xpu_place = paddle.XPUPlace(0)
        model_xpu = self.model.xpu(xpu_place)
        for name, p in model_xpu.named_parameters():
            self.assertTrue(
                p.place.is_xpu_place(), f"{name} not moved to XPU (XPUPlace)"
            )
        for name, b in model_xpu.named_buffers():
            self.assertTrue(
                b.place.is_xpu_place(),
                f"{name} buffer not moved to XPU (XPUPlace)",
            )

        with self.assertRaises(TypeError):
            self.model.xpu("xpu:0")

    def test_cpu(self):
        model_cpu = self.model.cpu()
        for name, p in model_cpu.named_parameters():
            self.assertTrue(p.place.is_cpu_place(), f"{name} not on CPU")
        for name, b in model_cpu.named_buffers():
            self.assertTrue(b.place.is_cpu_place(), f"{name} buffer not on CPU")


class TestType(unittest.TestCase):
    def setUp(self):
        self.module = Module()
        self.module.linear = nn.Linear(3, 2)
        self.module.bn = nn.BatchNorm1D(2)
        self.module.register_parameter(
            'weight',
            paddle.create_parameter(
                shape=[5, 3],
                dtype=paddle.float32,
                default_initializer=nn.initializer.Constant(1.0),
            ),
        )
        self.module.register_buffer(
            'buffer', paddle.ones([3, 2], dtype=paddle.float32)
        )

    def test_type(self):
        self.module.type(paddle.float64)
        self.assertEqual(self.module.weight.dtype, paddle.float64)
        self.assertEqual(self.module.buffer.dtype, paddle.float64)

        self.module.type('int8')
        self.assertEqual(self.module.weight.dtype, paddle.int8)
        self.assertEqual(self.module.buffer.dtype, paddle.int8)
        for name, param in self.module.named_parameters():
            self.assertEqual(param.dtype, paddle.int8)
        for name, buf in self.module.named_buffers():
            self.assertEqual(buf.dtype, paddle.int8)

    def test_double(self):
        self.module.double()
        self.assertEqual(self.module.weight.dtype, paddle.float64)
        self.assertEqual(self.module.buffer.dtype, paddle.float64)

    def test_half(self):
        self.module.half()
        self.assertEqual(self.module.weight.dtype, paddle.float16)
        self.assertEqual(self.module.buffer.dtype, paddle.float16)

    def test_type_error(self):
        with self.assertRaises(ValueError):
            self.module.type("invalid_dtype")


class TestStateDict(unittest.TestCase):
    def setUp(self):
        self.model = NestedModule()
        self.model1 = SubModule()

    def test_state_dict_basic(self):
        state = self.model.state_dict()
        self.assertIsInstance(state, OrderedDict)
        self.assertGreater(len(state), 0)

    def test_state_dict_contains_buffers(self):
        state = self.model.state_dict()
        buffer_keys = [k for k in state.keys() if "running_mean" in k]
        self.assertTrue(len(buffer_keys) > 0)
        for k in buffer_keys:
            self.assertIn("running_mean", k)
            self.assertTrue(isinstance(state[k], paddle.Tensor))

    def test_state_dict_prefix_structure(self):
        state = self.model.state_dict()
        expected_prefixes = [
            "layer1.linear.weight",
            "layer2.linear.weight",
            "layer3.linear.weight",
            "final_layer.weight",
        ]
        for prefix in expected_prefixes:
            match = [k for k in state.keys() if k.startswith(prefix)]
            self.assertTrue(len(match) > 0, f"Missing prefix: {prefix}")

    def test_state_dict_keep_vars(self):
        state1 = self.model1.state_dict(keep_vars=True)
        state2 = self.model1.state_dict(keep_vars=False)

        for k in state1.keys():
            if hasattr(state1[k], "stop_gradient"):
                self.assertFalse(state1[k].stop_gradient)
                self.assertTrue(state2[k].stop_gradient)

    def test_state_dict_with_positional_args(self):
        sd_default = self.model.state_dict()
        self.assertIsInstance(sd_default, dict)

        dest = {}
        sd1 = self.model.state_dict(dest)
        self.assertIsInstance(sd1, dict)

        sd2 = self.model.state_dict({}, "wfs")
        self.assertIsInstance(sd2, dict)

        sd3 = self.model.state_dict({}, False, "wfs")
        self.assertIsInstance(sd3, dict)

        sd4 = self.model.state_dict({}, "wfs", False)
        self.assertIsInstance(sd4, dict)

        sd5 = self.model.state_dict({}, False, "wfs", False, False)
        self.assertIsInstance(sd5, dict)


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.model = NestedModule()

    def test_train_sets_training_true(self):
        self.model.train(True)
        self.assertTrue(self.model.training)
        for name, submodule in self.model.named_children():
            self.assertTrue(submodule.training, f"{name} not in training mode")

    def test_train_sets_training_false(self):
        self.model.train(False)
        self.assertFalse(self.model.training)
        for name, submodule in self.model.named_children():
            self.assertFalse(submodule.training, f"{name} not in eval mode")

    def test_train_invalid_argument(self):
        with self.assertRaises(ValueError):
            self.model.train("True")


class TestGrad(unittest.TestCase):
    def setUp(self):
        self.model = SubModule1(10, 5)

    def test_requires_grad(self):
        for p in self.model.parameters():
            self.assertFalse(p.stop_gradient)

        self.model.requires_grad_(False)
        for p in self.model.parameters():
            self.assertTrue(p.stop_gradient)

        self.model.requires_grad_(True)
        for p in self.model.parameters():
            self.assertFalse(p.stop_gradient)

    def test_zero_grad(self):
        x = paddle.randn([4, 10])
        y = self.model(x).sum()
        y.backward()

        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)

        self.model.zero_grad(set_to_none=False)
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(paddle.allclose(p.grad, paddle.zeros_like(p.grad)))

        self.model.zero_grad()


# test ModuleList
class TestModuleListBasic(unittest.TestCase):
    def test_initialization_empty(self):
        module_list = nn.ModuleList()
        self.assertEqual(len(module_list), 0)
        self.assertEqual(list(module_list), [])

    def test_initialization_with_modules(self):
        modules = [nn.Linear(10, 5), nn.ReLU(), nn.Dropout(0.5)]
        module_list = nn.ModuleList(modules)
        self.assertEqual(len(module_list), 3)
        self.assertIsInstance(module_list[0], nn.Linear)
        self.assertIsInstance(module_list[1], nn.ReLU)
        self.assertIsInstance(module_list[2], nn.Dropout)

    def test_get_abs_string_index(self):
        modules = [nn.Linear(10, 5), nn.ReLU(), nn.Dropout(0.5)]
        module_list = nn.ModuleList(modules)
        self.assertEqual(module_list._get_abs_string_index(0), "0")
        self.assertEqual(module_list._get_abs_string_index(1), "1")
        self.assertEqual(module_list._get_abs_string_index(-1), "2")
        self.assertEqual(module_list._get_abs_string_index(-2), "1")

    def test_get_abs_string_index_out_of_range(self):
        module_list = nn.ModuleList([nn.Linear(10, 5)])

        with self.assertRaises(IndexError):
            module_list._get_abs_string_index(1)

        with self.assertRaises(IndexError):
            module_list._get_abs_string_index(-2)


class TestModuleListDir(unittest.TestCase):
    def test_dir_filters_numeric_keys(self):
        module_list = nn.ModuleList([nn.Linear(10, 5), nn.ReLU()])
        module_list.extra = nn.Linear(5, 2)
        d = dir(module_list)
        self.assertIn("extra", d)
        self.assertNotIn("0", d)
        self.assertNotIn("1", d)


class TestModuleListIndexing(unittest.TestCase):
    def setUp(self):
        self.modules = [nn.Linear(10, 5), nn.ReLU(), nn.Dropout(0.5), nn.Tanh()]
        self.module_list = nn.ModuleList(self.modules)

    def test_getitem_int(self):
        self.assertIs(self.module_list[0], self.modules[0])
        self.assertIs(self.module_list[1], self.modules[1])
        self.assertIs(self.module_list[-1], self.modules[3])
        self.assertIs(self.module_list[-2], self.modules[2])

    def test_getitem_slice(self):
        # Test basic slicing
        slice_result = self.module_list[1:3]
        self.assertIsInstance(slice_result, nn.ModuleList)
        self.assertEqual(len(slice_result), 2)
        self.assertIs(slice_result[0], self.modules[1])
        self.assertIs(slice_result[1], self.modules[2])

        # Test step slicing
        slice_step = self.module_list[::2]
        self.assertEqual(len(slice_step), 2)
        self.assertIs(slice_step[0], self.modules[0])
        self.assertIs(slice_step[1], self.modules[2])

        # Test negative slicing
        slice_neg = self.module_list[-2:]
        self.assertEqual(len(slice_neg), 2)
        self.assertIs(slice_neg[0], self.modules[2])
        self.assertIs(slice_neg[1], self.modules[3])

    def test_setitem(self):
        new_module = nn.Sigmoid()
        self.module_list[1] = new_module
        self.assertIs(self.module_list[1], new_module)
        self.assertIsNot(self.module_list[1], self.modules[1])

    def test_setitem_negative_index(self):
        new_module = nn.ELU()
        self.module_list[-1] = new_module
        self.assertIs(self.module_list[3], new_module)

    def test_delitem_int(self):
        original_length = len(self.module_list)
        del self.module_list[1]

        self.assertEqual(len(self.module_list), original_length - 1)
        self.assertIs(self.module_list[0], self.modules[0])
        self.assertIs(self.module_list[1], self.modules[2])
        self.assertIs(self.module_list[2], self.modules[3])

    def test_delitem_slice(self):
        # Delete middle slice
        del self.module_list[1:3]
        self.assertEqual(len(self.module_list), 2)
        self.assertIs(self.module_list[0], self.modules[0])
        self.assertIs(self.module_list[1], self.modules[3])

        # Reset and test deleting from start
        self.module_list = nn.ModuleList(self.modules)
        del self.module_list[:2]
        self.assertEqual(len(self.module_list), 2)
        self.assertIs(self.module_list[0], self.modules[2])
        self.assertIs(self.module_list[1], self.modules[3])

    def test_delitem_negative_index(self):
        original_length = len(self.module_list)
        del self.module_list[-1]
        self.assertEqual(len(self.module_list), original_length - 1)
        self.assertIs(self.module_list[-1], self.modules[2])


class TestModuleListOperations(unittest.TestCase):
    def setUp(self):
        self.modules1 = [nn.Linear(10, 5), nn.ReLU()]
        self.modules2 = [nn.Dropout(0.5), nn.Tanh()]
        self.module_list1 = nn.ModuleList(self.modules1)
        self.module_list2 = nn.ModuleList(self.modules2)

    def test_len(self):
        self.assertEqual(len(self.module_list1), 2)
        self.assertEqual(len(self.module_list2), 2)

    def test_iter(self):
        modules_from_iter = list(iter(self.module_list1))
        self.assertEqual(len(modules_from_iter), 2)
        self.assertIs(modules_from_iter[0], self.modules1[0])
        self.assertIs(modules_from_iter[1], self.modules1[1])

    def test_iadd(self):
        original_len = len(self.module_list1)
        self.module_list1 += self.modules2

        self.assertEqual(
            len(self.module_list1), original_len + len(self.modules2)
        )
        self.assertIs(self.module_list1[0], self.modules1[0])
        self.assertIs(self.module_list1[1], self.modules1[1])
        self.assertIs(self.module_list1[2], self.modules2[0])
        self.assertIs(self.module_list1[3], self.modules2[1])

    def test_add(self):
        combined = self.module_list1 + self.module_list2
        self.assertIsInstance(combined, nn.ModuleList)
        self.assertEqual(len(combined), 4)
        self.assertIs(combined[0], self.modules1[0])
        self.assertIs(combined[1], self.modules1[1])
        self.assertIs(combined[2], self.modules2[0])
        self.assertIs(combined[3], self.modules2[1])

        # Original lists should be unchanged
        self.assertEqual(len(self.module_list1), 2)
        self.assertEqual(len(self.module_list2), 2)

    def test_insert(self):
        new_module = nn.Sigmoid()
        self.module_list1.insert(1, new_module)

        self.assertEqual(len(self.module_list1), 3)
        self.assertIs(self.module_list1[0], self.modules1[0])
        self.assertIs(self.module_list1[1], new_module)
        self.assertIs(self.module_list1[2], self.modules1[1])

    def test_insert_at_beginning(self):
        new_module = nn.ELU()
        self.module_list1.insert(0, new_module)
        self.assertIs(self.module_list1[0], new_module)
        self.assertIs(self.module_list1[1], self.modules1[0])

    def test_insert_at_end(self):
        new_module = nn.LeakyReLU()
        self.module_list1.insert(2, new_module)
        self.assertIs(self.module_list1[2], new_module)

    def test_append(self):
        new_module = nn.Softmax()
        original_len = len(self.module_list1)
        result = self.module_list1.append(new_module)

        self.assertEqual(len(self.module_list1), original_len + 1)
        self.assertIs(self.module_list1[-1], new_module)
        self.assertIs(result, self.module_list1)

    def test_pop_int(self):
        popped = self.module_list1.pop(0)
        self.assertIs(popped, self.modules1[0])
        self.assertEqual(len(self.module_list1), 1)
        self.assertIs(self.module_list1[0], self.modules1[1])

    def test_pop_negative_index(self):
        popped = self.module_list1.pop(-1)
        self.assertIs(popped, self.modules1[1])
        self.assertEqual(len(self.module_list1), 1)

    def test_pop_slice(self):
        modules = [nn.Linear(10, 5), nn.ReLU(), nn.Dropout(0.5), nn.Tanh()]
        module_list = nn.ModuleList(modules)

        popped = module_list.pop(slice(1, 3))
        self.assertIsInstance(popped, nn.ModuleList)
        self.assertEqual(len(popped), 2)
        self.assertIs(popped[0], modules[1])
        self.assertIs(popped[1], modules[2])
        self.assertEqual(len(module_list), 2)

    def test_extend(self):
        additional_modules = [nn.Dropout(0.3), nn.Sigmoid()]
        original_len = len(self.module_list1)
        result = self.module_list1.extend(additional_modules)

        self.assertEqual(
            len(self.module_list1), original_len + len(additional_modules)
        )
        self.assertIs(self.module_list1[2], additional_modules[0])
        self.assertIs(self.module_list1[3], additional_modules[1])
        self.assertIs(result, self.module_list1)

    def test_extent_error(self):
        with self.assertRaises(TypeError):
            self.module_list1.extend(123)


class TestModuleListFunctionality(unittest.TestCase):
    def test_module_list_parameters(self):
        linear1 = nn.Linear(10, 5)
        linear2 = nn.Linear(5, 2)
        module_list = nn.ModuleList([linear1, linear2])

        params = dict(module_list.named_parameters())
        self.assertIn('0.weight', params)
        self.assertIn('0.bias', params)
        self.assertIn('1.weight', params)
        self.assertIn('1.bias', params)

    def test_module_list_forward(self):
        class TestModule(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = TestModule()
        x = paddle.ones([2, 10])
        output = model(x)

        self.assertEqual(output.shape, [2, 2])
        self.assertIsInstance(output, paddle.Tensor)

    def test_module_list_state_dict(self):
        linear1 = nn.Linear(10, 5)
        linear2 = nn.Linear(5, 2)
        module_list = nn.ModuleList([linear1, linear2])

        state_dict = module_list.state_dict()
        expected_keys = {'0.weight', '0.bias', '1.weight', '1.bias'}
        self.assertEqual(set(state_dict.keys()), expected_keys)

    def test_module_list_gradients(self):
        linear1 = nn.Linear(10, 5)
        linear2 = nn.Linear(5, 1)
        module_list = nn.ModuleList([linear1, linear2])

        x = paddle.ones([1, 10])
        y = module_list[1](module_list[0](x))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(module_list[0].weight.grad)
        self.assertIsNotNone(module_list[1].weight.grad)


class TestModuleListEdgeCases(unittest.TestCase):
    def test_empty_operations(self):
        empty_list = nn.ModuleList()

        self.assertEqual(len(empty_list), 0)
        self.assertEqual(list(empty_list), [])

        module = nn.Linear(5, 3)
        empty_list.append(module)
        self.assertEqual(len(empty_list), 1)
        self.assertIs(empty_list[0], module)

    def test_single_element_operations(self):
        module = nn.ReLU()
        single_list = nn.ModuleList([module])

        self.assertEqual(len(single_list), 1)
        self.assertIs(single_list[0], module)

        del single_list[0]
        self.assertEqual(len(single_list), 0)

    def test_duplicate_modules(self):
        shared_module = nn.Dropout(0.5)
        module_list = nn.ModuleList([shared_module, shared_module])

        self.assertEqual(len(module_list), 2)
        self.assertIs(module_list[0], module_list[1])

    def test_module_list_with_none(self):
        module_list = nn.ModuleList([nn.Linear(5, 3), None, nn.ReLU()])
        self.assertEqual(len(module_list), 3)
        self.assertIsNone(module_list[1])


# test ModuleDict
class TestModuleDictBasic(unittest.TestCase):
    def test_initialization_empty(self):
        module_dict = nn.ModuleDict()
        self.assertEqual(len(module_dict), 0)
        self.assertEqual(list(module_dict), [])

    def test_initialization_with_modules(self):
        modules = {
            'linear': nn.Linear(10, 5),
            'activation': nn.ReLU(),
            'dropout': nn.Dropout(0.5),
        }
        module_dict = nn.ModuleDict(modules)

        self.assertEqual(len(module_dict), 3)
        self.assertIsInstance(module_dict['linear'], nn.Linear)
        self.assertIsInstance(module_dict['activation'], nn.ReLU)
        self.assertIsInstance(module_dict['dropout'], nn.Dropout)

    def test_initialization_with_ordered_dict(self):
        modules = OrderedDict(
            [
                ('conv', nn.Conv2D(3, 16, 3)),
                ('bn', nn.BatchNorm2D(16)),
                ('pool', nn.MaxPool2D(2)),
            ]
        )
        module_dict = nn.ModuleDict(modules)

        self.assertEqual(len(module_dict), 3)
        self.assertIsInstance(module_dict['conv'], nn.Conv2D)
        self.assertIsInstance(module_dict['bn'], nn.BatchNorm2D)
        self.assertIsInstance(module_dict['pool'], nn.MaxPool2D)


class TestModuleDictAccessMethods(unittest.TestCase):
    def setUp(self):
        self.modules = {
            'linear1': nn.Linear(10, 5),
            'relu': nn.ReLU(),
            'linear2': nn.Linear(5, 2),
            'sigmoid': nn.Sigmoid(),
        }
        self.module_dict = nn.ModuleDict(self.modules)

    def test_getitem(self):
        self.assertIs(self.module_dict['linear1'], self.modules['linear1'])
        self.assertIs(self.module_dict['relu'], self.modules['relu'])
        self.assertIs(self.module_dict['linear2'], self.modules['linear2'])

    def test_getitem_key_error(self):
        with self.assertRaises(KeyError):
            _ = self.module_dict['nonexistent']

    def test_setitem(self):
        new_module = nn.Tanh()
        self.module_dict['tanh'] = new_module
        self.assertIs(self.module_dict['tanh'], new_module)
        self.assertEqual(len(self.module_dict), 5)

    def test_setitem_overwrite(self):
        new_linear = nn.Linear(10, 8)
        original_length = len(self.module_dict)
        self.module_dict['linear1'] = new_linear

        self.assertEqual(len(self.module_dict), original_length)
        self.assertIs(self.module_dict['linear1'], new_linear)
        self.assertIsNot(self.module_dict['linear1'], self.modules['linear1'])

    def test_delitem(self):
        original_length = len(self.module_dict)
        del self.module_dict['relu']

        self.assertEqual(len(self.module_dict), original_length - 1)
        self.assertNotIn('relu', self.module_dict)
        self.assertIn('linear1', self.module_dict)
        self.assertIn('linear2', self.module_dict)

    def test_delitem_key_error(self):
        with self.assertRaises(KeyError):
            del self.module_dict['nonexistent']

    def test_contains(self):
        self.assertIn('linear1', self.module_dict)
        self.assertIn('relu', self.module_dict)
        self.assertNotIn('nonexistent', self.module_dict)

    def test_keys(self):
        keys = list(self.module_dict.keys())
        expected_keys = ['linear1', 'relu', 'linear2', 'sigmoid']
        self.assertEqual(set(keys), set(expected_keys))
        self.assertEqual(len(keys), len(expected_keys))

    def test_values(self):
        values = list(self.module_dict.values())
        self.assertEqual(len(values), 4)
        self.assertIn(self.modules['linear1'], values)
        self.assertIn(self.modules['relu'], values)
        self.assertIn(self.modules['linear2'], values)

    def test_items(self):
        items = list(self.module_dict.items())
        self.assertEqual(len(items), 4)

        keys, values = zip(*items)
        self.assertEqual(set(keys), set(self.modules.keys()))
        self.assertEqual(set(values), set(self.modules.values()))

    def test_iter(self):
        keys_from_iter = list(iter(self.module_dict))
        expected_keys = ['linear1', 'relu', 'linear2', 'sigmoid']
        self.assertEqual(set(keys_from_iter), set(expected_keys))


class TestModuleDictOperations(unittest.TestCase):
    def setUp(self):
        self.initial_modules = {'conv': nn.Conv2D(3, 16, 3), 'relu': nn.ReLU()}
        self.module_dict = nn.ModuleDict(self.initial_modules)

    def test_clear(self):
        self.assertEqual(len(self.module_dict), 2)
        self.module_dict.clear()
        self.assertEqual(len(self.module_dict), 0)
        self.assertEqual(list(self.module_dict.keys()), [])

    def test_pop(self):
        original_length = len(self.module_dict)
        popped_module = self.module_dict.pop('conv')

        self.assertEqual(len(self.module_dict), original_length - 1)
        self.assertIs(popped_module, self.initial_modules['conv'])
        self.assertNotIn('conv', self.module_dict)
        self.assertIn('relu', self.module_dict)

    def test_pop_key_error(self):
        with self.assertRaises(KeyError):
            self.module_dict.pop('nonexistent')

    def test_update_with_dict(self):
        new_modules = {'bn': nn.BatchNorm2D(16), 'pool': nn.MaxPool2D(2)}
        self.module_dict.update(new_modules)

        self.assertEqual(len(self.module_dict), 4)
        self.assertIn('conv', self.module_dict)
        self.assertIn('relu', self.module_dict)
        self.assertIn('bn', self.module_dict)
        self.assertIn('pool', self.module_dict)

    def test_update_with_ordered_dict(self):
        new_modules = OrderedDict(
            [('dropout', nn.Dropout(0.5)), ('linear', nn.Linear(16, 10))]
        )
        self.module_dict.update(new_modules)

        self.assertEqual(len(self.module_dict), 4)
        self.assertIn('dropout', self.module_dict)
        self.assertIn('linear', self.module_dict)

    def test_update_with_module_dict(self):
        other_dict = nn.ModuleDict({'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()})
        self.module_dict.update(other_dict)

        self.assertEqual(len(self.module_dict), 4)
        self.assertIn('sigmoid', self.module_dict)
        self.assertIn('tanh', self.module_dict)

    def test_update_with_iterable(self):
        new_modules = [('bn', nn.BatchNorm2D(16)), ('pool', nn.MaxPool2D(2))]
        self.module_dict.update(new_modules)

        self.assertEqual(len(self.module_dict), 4)
        self.assertIn('bn', self.module_dict)
        self.assertIn('pool', self.module_dict)

    def test_update_overwrite_existing(self):
        new_conv = nn.Conv2D(3, 32, 3)
        self.module_dict.update({'conv': new_conv})

        self.assertEqual(len(self.module_dict), 2)
        self.assertIs(self.module_dict['conv'], new_conv)
        self.assertIsNot(self.module_dict['conv'], self.initial_modules['conv'])

    def test_update_invalid_iterable_element(self):
        with self.assertRaises(TypeError):
            self.module_dict.update([nn.Linear(5, 3)])

    def test_update_invalid_pair_length(self):
        with self.assertRaises(ValueError):
            self.module_dict.update([('key', 'module', 'extra')])

    def test_update_error(self):
        with self.assertRaises(AssertionError):
            self.module_dict.update(123)


class TestModuleDictFunctionality(unittest.TestCase):
    def test_module_dict_parameters(self):
        module_dict = nn.ModuleDict(
            {'linear1': nn.Linear(10, 5), 'linear2': nn.Linear(5, 2)}
        )

        params = dict(module_dict.named_parameters())
        self.assertIn('linear1.weight', params)
        self.assertIn('linear1.bias', params)
        self.assertIn('linear2.weight', params)
        self.assertIn('linear2.bias', params)

    def test_module_dict_forward(self):
        class TestModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict(
                    {
                        'linear1': nn.Linear(10, 5),
                        'activation': nn.ReLU(),
                        'linear2': nn.Linear(5, 2),
                    }
                )

            def forward(self, x):
                x = self.layers['linear1'](x)
                x = self.layers['activation'](x)
                x = self.layers['linear2'](x)
                return x

        model = TestModel()
        x = paddle.ones([2, 10])
        output = model(x)

        self.assertEqual(output.shape, [2, 2])
        self.assertIsInstance(output, paddle.Tensor)

    def test_module_dict_state_dict(self):
        module_dict = nn.ModuleDict(
            {'conv': nn.Conv2D(3, 16, 3), 'bn': nn.BatchNorm2D(16)}
        )

        state_dict = module_dict.state_dict()
        expected_keys = {
            'conv.weight',
            'conv.bias',
            'bn.weight',
            'bn.bias',
            'bn._mean',
            'bn._variance',
        }
        actual_keys = set(state_dict.keys())
        self.assertTrue(expected_keys.issubset(actual_keys))

    def test_module_dict_gradients(self):
        module_dict = nn.ModuleDict(
            {'linear1': nn.Linear(10, 5), 'linear2': nn.Linear(5, 1)}
        )

        x = paddle.ones([1, 10])
        y = module_dict['linear2'](module_dict['linear1'](x))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(module_dict['linear1'].weight.grad)
        self.assertIsNotNone(module_dict['linear2'].weight.grad)


class TestModuleDictEdgeCases(unittest.TestCase):
    def test_empty_operations(self):
        empty_dict = nn.ModuleDict()

        self.assertEqual(len(empty_dict), 0)
        self.assertEqual(list(empty_dict.keys()), [])
        self.assertEqual(list(empty_dict.values()), [])

        module = nn.Linear(5, 3)
        empty_dict['linear'] = module
        self.assertEqual(len(empty_dict), 1)
        self.assertIs(empty_dict['linear'], module)

    def test_single_element_operations(self):
        module = nn.ReLU()
        single_dict = nn.ModuleDict({'activation': module})

        self.assertEqual(len(single_dict), 1)
        self.assertIs(single_dict['activation'], module)

        del single_dict['activation']
        self.assertEqual(len(single_dict), 0)

    def test_duplicate_modules(self):
        shared_module = nn.Dropout(0.5)
        module_dict = nn.ModuleDict(
            {'dropout1': shared_module, 'dropout2': shared_module}
        )

        self.assertEqual(len(module_dict), 2)
        self.assertIs(module_dict['dropout1'], module_dict['dropout2'])

    def test_special_key_names(self):
        module_dict = nn.ModuleDict()
        module_dict['key-with-dash'] = nn.Linear(5, 3)
        module_dict['key_with_underscore'] = nn.ReLU()
        module_dict['key.with.dots'] = nn.Sigmoid()
        module_dict['123numeric'] = nn.Tanh()

        self.assertEqual(len(module_dict), 4)
        self.assertIn('key-with-dash', module_dict)
        self.assertIn('key_with_underscore', module_dict)
        self.assertIn('key.with.dots', module_dict)
        self.assertIn('123numeric', module_dict)

    def test_module_dict_with_none(self):
        module_dict = nn.ModuleDict(
            {
                'linear': nn.Linear(5, 3),
                'none_module': None,
                'activation': nn.ReLU(),
            }
        )
        self.assertEqual(len(module_dict), 3)
        self.assertIsNone(module_dict['none_module'])

    def test_key_ordering(self):
        modules = OrderedDict(
            [
                ('z_last', nn.Linear(5, 3)),
                ('a_first', nn.ReLU()),
                ('m_middle', nn.Sigmoid()),
            ]
        )
        module_dict = nn.ModuleDict(modules)

        keys = list(module_dict.keys())
        self.assertEqual(keys, ['z_last', 'a_first', 'm_middle'])


class TestModuleDictIntegration(unittest.TestCase):
    def test_combined_with_module_list(self):
        linear_layers = nn.ModuleList([nn.Linear(10, 5), nn.Linear(5, 2)])
        activations = nn.ModuleDict(
            {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        )

        class CombinedModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.layers = linear_layers
                self.activations = activations

            def forward(self, x):
                x = self.layers[0](x)
                x = self.activations['relu'](x)
                x = self.layers[1](x)
                x = self.activations['sigmoid'](x)
                return x

        model = CombinedModel()
        x = paddle.ones([2, 10])
        output = model(x)

        self.assertEqual(output.shape, [2, 2])
        self.assertIsInstance(output, paddle.Tensor)


if __name__ == '__main__':
    unittest.main()
