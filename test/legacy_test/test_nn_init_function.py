#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import random
import unittest

import numpy as np
from scipy import stats

import paddle
from paddle import nn
from paddle.pir.core import ParameterMeta

DELTA = 0.00001


class TestKaimingUniformFunc(unittest.TestCase):
    def _test_kaiming_uniform_common(self, tensor):
        init = paddle.nn.init.kaiming_uniform_
        init(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu")
        init(tensor, a=-0.2, mode="fan_out", nonlinearity="leaky_relu")
        init(tensor, a=0, mode="fan_in", nonlinearity="relu")
        init(tensor, a=0, mode="fan_out", nonlinearity="relu")

    def test_kaiming_uniform_linear(self):
        linear = nn.Linear(40, 20)
        self._test_kaiming_uniform_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view([-1]).tolist()
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def calculate_gain(self, nonlinearity, param):
        recommended_gain = {
            'sigmoid': 1,
            'linear': 1,
            'conv1d': 1,
            'conv2d': 1,
            'conv3d': 1,
            'conv1d_transpose': 1,
            'conv_transpose1d': 1,
            'conv2d_transpose': 1,
            'conv_transpose2d': 1,
            'conv3d_transpose': 1,
            'conv_transpose3d': 1,
            'tanh': 5.0 / 3,
            'relu': math.sqrt(2.0),
            'leaky_relu': math.sqrt(2.0 / (1 + param**2)),
            'selu': 3.0 / 4,
        }
        return recommended_gain[nonlinearity]

    def test_kaiming_uniform_nonlinearity(self):
        for nonlinearity in [
            'conv_transpose1d',
            'conv_transpose2d',
            'conv_transpose3d',
            'relu',
            'leaky_relu',
        ]:
            input_tensor = paddle.zeros([1024, 512])
            paddle.nn.init.kaiming_uniform_(
                input_tensor, nonlinearity=nonlinearity
            )

            fan_in = input_tensor.shape[0]

            expected_std = self.calculate_gain(
                nonlinearity=nonlinearity, param=0
            )

            bounds = expected_std * math.sqrt(3.0 / float(fan_in))
            assert self._is_uniform(input_tensor, -bounds, bounds)

    def test_kaiming_uniform(self):
        for use_a in [True, False]:
            for dims in [2, 3, 4]:
                for mode in ["fan_in", "fan_out"]:
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=108
                    )
                    if use_a:
                        a = self._random_float(0.1, 2)
                        paddle.nn.init.kaiming_uniform_(
                            input_tensor, a=a, mode=mode
                        )
                    else:
                        a = 0
                        paddle.nn.init.kaiming_uniform_(input_tensor, mode=mode)

                    if dims == 2:
                        # This is the case for simple matrix multiply
                        fan_in = input_tensor.shape[0]
                        fan_out = input_tensor.shape[1]
                    else:
                        fan_in = input_tensor.shape[1]
                        fan_out = input_tensor.shape[0]

                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out
                    expected_std = self.calculate_gain(
                        nonlinearity='leaky_relu', param=a
                    )
                    bounds = expected_std * math.sqrt(3.0 / float(n))
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_kaiming_uniform_fp16(self):
        input_tensor = paddle.zeros([1024, 512], dtype='float16')
        paddle.nn.init.kaiming_uniform_(input_tensor)
        fan_in = input_tensor.shape[0]

        expected_std = self.calculate_gain(nonlinearity='leaky_relu', param=0)

        bounds = expected_std * math.sqrt(3.0 / float(fan_in))
        assert self._is_uniform(input_tensor, -bounds, bounds)
        assert input_tensor.dtype == paddle.float16


class TestKaimingUniformFuncPir(unittest.TestCase):
    def setUp(self):
        self.init_uniform_op_name = 'pd_op.uniform'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_kaiming_uniform_(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([1024, 512], paddle.float32)
                init_result = paddle.nn.init.kaiming_uniform_(parameter_meta)
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(6.0 / init_result.shape[0])

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_kaiming_uniform_conv(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.kaiming_uniform_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    6.0
                    / (
                        init_result.shape[1]
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_kaiming_uniform_fan_out(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.kaiming_uniform_(
                    parameter_meta,
                    mode='fan_out',
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    6.0
                    / (
                        init_result.shape[0]
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)


class TestKaimingNormalFunc(unittest.TestCase):
    def _test_kaiming_normal_common(self, tensor):
        init = paddle.nn.init.kaiming_normal_
        init(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu")
        init(tensor, a=-0.2, mode="fan_out", nonlinearity="leaky_relu")
        init(tensor, a=0, mode="fan_in", nonlinearity="relu")
        init(tensor, a=0, mode="fan_out", nonlinearity="relu")

    def test_kaiming_normal_linear(self):
        linear = nn.Linear(40, 20)
        self._test_kaiming_normal_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_normal(self, tensor, mean, std):
        samples = tensor.view([-1]).tolist()
        p_value = stats.kstest(samples, "norm", args=(mean, std))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def calculate_gain(self, nonlinearity, param):
        recommended_gain = {
            'sigmoid': 1,
            'linear': 1,
            'conv1d': 1,
            'conv2d': 1,
            'conv3d': 1,
            'conv1d_transpose': 1,
            'conv_transpose1d': 1,
            'conv2d_transpose': 1,
            'conv_transpose2d': 1,
            'conv3d_transpose': 1,
            'conv_transpose3d': 1,
            'tanh': 5.0 / 3,
            'relu': math.sqrt(2.0),
            'leaky_relu': math.sqrt(2.0 / (1 + param**2)),
            'selu': 3.0 / 4,
        }
        return recommended_gain[nonlinearity]

    def test_kaiming_normal_nonlinearity(self):
        for nonlinearity in [
            'conv_transpose1d',
            'conv_transpose2d',
            'conv_transpose3d',
            'relu',
            'leaky_relu',
        ]:
            input_tensor = paddle.zeros([1024, 512])
            paddle.nn.init.kaiming_normal_(
                input_tensor, nonlinearity=nonlinearity
            )

            fan_in = input_tensor.shape[0]

            expected_std = self.calculate_gain(
                nonlinearity=nonlinearity, param=0
            )

            std = expected_std / math.sqrt(float(fan_in))
            assert self._is_normal(input_tensor, 0.0, std)

    def test_kaiming_normal(self):
        for use_a in [True, False]:
            for dims in [2, 3, 4]:
                for mode in ["fan_in", "fan_out"]:
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=108
                    )
                    if use_a:
                        a = self._random_float(0.1, 2)
                        paddle.nn.init.kaiming_normal_(
                            input_tensor, a=a, mode=mode
                        )
                    else:
                        a = 0
                        paddle.nn.init.kaiming_normal_(input_tensor, mode=mode)

                    if dims == 2:
                        # This is the case for simple matrix multiply
                        fan_in = input_tensor.shape[0]
                        fan_out = input_tensor.shape[1]
                    else:
                        fan_in = input_tensor.shape[1]
                        fan_out = input_tensor.shape[0]

                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out
                    expected_std = self.calculate_gain(
                        nonlinearity='leaky_relu', param=a
                    )
                    std = expected_std / math.sqrt(float(n))
                    assert self._is_normal(input_tensor, 0.0, std)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_kaiming_normal_fp16(self):
        input_tensor = paddle.zeros([1024, 512], dtype='float16')
        paddle.nn.init.kaiming_normal_(input_tensor)
        fan_in = input_tensor.shape[0]

        expected_std = self.calculate_gain(nonlinearity='leaky_relu', param=0)

        std = expected_std / math.sqrt(float(fan_in))
        assert self._is_normal(input_tensor, 0.0, std)
        assert input_tensor.dtype == paddle.float16


class TestKaimingNormalFuncPir(unittest.TestCase):
    def setUp(self):
        self.init_normal_op_name = 'pd_op.gaussian'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_kaiming_normal_(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([1024, 512], paddle.float32)
                init_result = paddle.nn.init.kaiming_normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(2.0 / init_result.shape[0])
                self.assertAlmostEqual(
                    init_op.attrs()["mean"], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(
                    init_op.attrs()["std"], limit, delta=DELTA
                )
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_kaiming_normal_conv(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.kaiming_normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    2.0
                    / (
                        init_result.shape[1]
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                self.assertAlmostEqual(
                    init_op.attrs()["mean"], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(
                    init_op.attrs()["std"], limit, delta=DELTA
                )
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_kaiming_normal_fan_out(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.kaiming_normal_(
                    parameter_meta,
                    mode='fan_out',
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    2.0
                    / (
                        init_result.shape[0]
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                self.assertAlmostEqual(
                    init_op.attrs()["mean"], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(
                    init_op.attrs()["std"], limit, delta=DELTA
                )
                self.assertEqual(init_op.attrs()['seed'], 0)


class TestXavierUniformFunc(unittest.TestCase):
    def _test_xavier_uniform_common(self, tensor):
        init = paddle.nn.init.xavier_uniform_
        init(tensor, gain=0.2)
        init(tensor, gain=0.25)
        init(tensor, gain=1.0)
        init(tensor, gain=2.0)

    def test_xavier_uniform_linear(self):
        linear = nn.Linear(40, 20)
        self._test_xavier_uniform_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view([-1]).tolist()
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_xavier_uniform(self):
        for use_gain in [True, False]:
            for dims in [2, 3, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                if use_gain:
                    gain = self._random_float(0.1, 3.0)
                else:
                    gain = 1.0
                paddle.nn.init.xavier_uniform_(input_tensor, gain=gain)

                if dims == 2:
                    # This is the case for simple matrix multiply
                    fan_in = input_tensor.shape[0]
                    fan_out = input_tensor.shape[1]
                else:
                    fan_in = input_tensor.shape[1]
                    fan_out = input_tensor.shape[0]

                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                bounds = gain * math.sqrt(6.0 / float(fan_in + fan_out))
                assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_xavier_uniform_fp16(self):
        input_tensor = paddle.zeros([1024, 512], dtype='float16')
        paddle.nn.init.xavier_uniform_(input_tensor)
        fan_in = input_tensor.shape[0]
        fan_out = input_tensor.shape[1]

        bounds = math.sqrt(6.0 / float(fan_in + fan_out))
        assert self._is_uniform(input_tensor, -bounds, bounds)
        assert input_tensor.dtype == paddle.float16


class TestXavierUniformFuncPir(unittest.TestCase):
    def setUp(self):
        self.init_uniform_op_name = 'pd_op.uniform'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_xavier_uniform_(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([1024, 512], paddle.float32)
                init_result = paddle.nn.init.xavier_uniform_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    6.0 / (init_result.shape[0] + init_result.shape[1])
                )

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_xavier_uniform_conv(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.xavier_uniform_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    6.0
                    / (
                        (init_result.shape[0] + init_result.shape[1])
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_xavier_uniform_gain(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.xavier_uniform_(
                    parameter_meta,
                    gain=0.5,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                limit = 0.5 * np.sqrt(
                    6.0
                    / (
                        (init_result.shape[0] + init_result.shape[1])
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)


class TestXavierNormalFunc(unittest.TestCase):
    def _test_xavier_normal_common(self, tensor):
        init = paddle.nn.init.xavier_normal_
        init(tensor, gain=0.2)
        init(tensor, gain=0.25)
        init(tensor, gain=1.0)
        init(tensor, gain=2.0)

    def test_xavier_normal_linear(self):
        linear = nn.Linear(40, 20)
        self._test_xavier_normal_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_normal(self, tensor, mean, std):
        samples = tensor.view([-1]).tolist()
        p_value = stats.kstest(samples, "norm", args=(mean, std))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_xavier_uniform(self):
        for use_gain in [True, False]:
            for dims in [2, 3, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                if use_gain:
                    gain = self._random_float(0.1, 3.0)
                else:
                    gain = 1.0
                paddle.nn.init.xavier_normal_(input_tensor, gain=gain)

                if dims == 2:
                    # This is the case for simple matrix multiply
                    fan_in = input_tensor.shape[0]
                    fan_out = input_tensor.shape[1]
                else:
                    fan_in = input_tensor.shape[1]
                    fan_out = input_tensor.shape[0]

                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                bounds = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                assert self._is_normal(input_tensor, 0.0, bounds)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_xavier_normal_fp16(self):
        input_tensor = paddle.zeros([1024, 512], dtype='float16')
        paddle.nn.init.xavier_normal_(input_tensor)
        fan_in = input_tensor.shape[0]
        fan_out = input_tensor.shape[1]

        bounds = math.sqrt(2.0 / float(fan_in + fan_out))
        assert self._is_normal(input_tensor, 0.0, bounds)
        assert input_tensor.dtype == paddle.float16


class TestXavierNormalFuncPir(unittest.TestCase):
    def setUp(self):
        self.init_normal_op_name = 'pd_op.gaussian'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_xavier_normal_(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([1024, 512], paddle.float32)
                init_result = paddle.nn.init.xavier_normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    2.0 / (init_result.shape[0] + init_result.shape[1])
                )

                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(
                    init_op.attrs()['std'], limit, delta=DELTA
                )
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_xavier_normal_conv(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.xavier_normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(
                    2.0
                    / (
                        (init_result.shape[0] + init_result.shape[1])
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(
                    init_op.attrs()['std'], limit, delta=DELTA
                )
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_xavier_normal_gain(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.xavier_normal_(
                    parameter_meta,
                    gain=0.5,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                limit = 0.5 * np.sqrt(
                    2.0
                    / (
                        (init_result.shape[0] + init_result.shape[1])
                        * init_result.shape[2]
                        * init_result.shape[3]
                    )
                )

                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(
                    init_op.attrs()['std'], limit, delta=DELTA
                )
                self.assertEqual(init_op.attrs()['seed'], 0)


class TestUniformFunc(unittest.TestCase):
    def _test_uniform_common(self, tensor):
        init = paddle.nn.init.uniform_
        init(tensor, a=0.2, b=1.3)
        init(tensor, a=2.2, b=4.3)
        init(tensor, a=-0.2, b=-0.2)
        init(tensor, a=-1.5, b=1.5)

    def test_uniform_linear(self):
        linear = nn.Linear(40, 20)
        self._test_uniform_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view([-1]).tolist()
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_uniform(self):
        for dims in [2, 3, 4]:
            input_tensor = self._create_random_nd_tensor(
                dims, size_min=20, size_max=108
            )
            paddle.nn.init.uniform_(input_tensor, a=-3.0, b=2.0)

            assert self._is_uniform(input_tensor, -3.0, 2.0)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_uniform_fp16(self):
        input_tensor = paddle.zeros([1024, 512], dtype='float16')
        paddle.nn.init.uniform_(input_tensor)

        assert self._is_uniform(input_tensor, 0.0, 1.0)
        assert input_tensor.dtype == paddle.float16


class TestUniformFuncPir(unittest.TestCase):
    def setUp(self):
        self.init_uniform_op_name = 'pd_op.uniform'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_uniform_(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([1024, 512], paddle.float32)
                init_result = paddle.nn.init.uniform_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, 0.0, delta=DELTA)
                self.assertAlmostEqual(max, 1.0, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_uniform_conv(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.uniform_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, 0.0, delta=DELTA)
                self.assertAlmostEqual(max, 1.0, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_uniform_bound(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.uniform_(
                    parameter_meta,
                    a=-0.8,
                    b=0.8,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -0.8, delta=DELTA)
                self.assertAlmostEqual(max, 0.8, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)


class TestNormalFunc(unittest.TestCase):
    def _test_normal_common(self, tensor):
        init = paddle.nn.init.normal_
        init(tensor, mean=0.2, std=2.0)
        init(tensor, mean=0.0, std=2.0)
        init(tensor, mean=-0.1, std=1.0)
        init(tensor, mean=0.4, std=2.5)

    def test_normal_linear(self):
        linear = nn.Linear(40, 20)
        self._test_normal_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_normal(self, tensor, mean, std):
        samples = tensor.view([-1]).tolist()
        p_value = stats.kstest(samples, "norm", args=(mean, std))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_normal(self):
        for dims in [2, 3, 4]:
            input_tensor = self._create_random_nd_tensor(
                dims, size_min=20, size_max=108
            )
            mean = self._random_float(-3.0, 3.0)
            std = self._random_float(0.5, 3.0)
            paddle.nn.init.normal_(input_tensor, mean=mean, std=std)

            assert self._is_normal(input_tensor, mean, std)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_normal_fp16(self):
        input_tensor = paddle.zeros([1024, 512], dtype='float16')
        paddle.nn.init.normal_(input_tensor)
        assert self._is_normal(input_tensor, 0.0, 1.0)
        assert input_tensor.dtype == paddle.float16


class TestNormalFuncPir(unittest.TestCase):
    def setUp(self):
        self.init_normal_op_name = 'pd_op.gaussian'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_normal_(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([1024, 512], paddle.float32)
                init_result = paddle.nn.init.normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], 1.0, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_conv(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], 1.0, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_mean_std(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.normal_(
                    parameter_meta,
                    mean=0.3,
                    std=3.5,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.3, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], 3.5, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)


class TestTruncNormalFunc(unittest.TestCase):
    def _test_normal_common(self, tensor):
        init = paddle.nn.init.trunc_normal_
        init(tensor, mean=0.2, std=2.0, a=-1.8, b=2.2)
        init(tensor, mean=0.0, std=2.0, a=-2, b=2)
        init(tensor, mean=-0.1, std=1.0, a=-2, b=2)
        init(tensor, mean=0.4, std=2.5, a=-3, b=3)

    def test_normal_linear(self):
        linear = nn.Linear(40, 20)
        self._test_normal_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_trunc_normal(self, tensor, mean, std, a, b):
        z_samples = (tensor.view([-1]) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, "truncnorm", args=(a0, b0))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_normal(self):
        for dims in [2, 3, 4]:
            input_tensor = self._create_random_nd_tensor(
                dims, size_min=20, size_max=108
            )
            mean = self._random_float(-3.0, 3.0)
            std = self._random_float(0.5, 3.0)
            bound = self._random_float(0.5, 10)
            a = mean - bound
            b = mean + bound
            paddle.nn.init.trunc_normal_(
                input_tensor, mean=mean, std=std, a=a, b=b
            )

            assert self._is_trunc_normal(input_tensor, mean, std, a=a, b=b)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_normal_fp16(self):
        input_tensor = paddle.zeros([1024, 512], dtype='float16')
        paddle.nn.init.trunc_normal_(input_tensor)
        assert self._is_trunc_normal(input_tensor, 0.0, 1.0, -2, 2)
        assert input_tensor.dtype == paddle.float16


class TestTruncNormalFuncPir(unittest.TestCase):
    def setUp(self):
        self.init_normal_op_name = 'pd_op.truncated_gaussian_random'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_trunc_normal_(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([1024, 512], paddle.float32)
                init_result = paddle.nn.init.trunc_normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], 1.0, delta=DELTA)
                self.assertAlmostEqual(init_op.attrs()['a'], -2.0, delta=DELTA)
                self.assertAlmostEqual(init_op.attrs()['b'], 2.0, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_conv(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.trunc_normal_(
                    parameter_meta,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], 1.0, delta=DELTA)
                self.assertAlmostEqual(init_op.attrs()['a'], -2.0, delta=DELTA)
                self.assertAlmostEqual(init_op.attrs()['b'], 2.0, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_mean_std(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                parameter_meta = ParameterMeta([5, 10, 15, 20], paddle.float32)
                init_result = paddle.nn.init.trunc_normal_(
                    parameter_meta,
                    mean=0.3,
                    std=3.5,
                    a=-3.0,
                    b=3.0,
                )
                block = main.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]

                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.3, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], 3.5, delta=DELTA)
                self.assertAlmostEqual(init_op.attrs()['a'], -3.0, delta=DELTA)
                self.assertAlmostEqual(init_op.attrs()['b'], 3.0, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)


if __name__ == '__main__':
    unittest.main()
