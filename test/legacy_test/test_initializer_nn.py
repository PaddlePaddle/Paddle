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

import unittest

import numpy as np
from utils import static_guard

import paddle
from paddle import base, nn
from paddle.base import framework
from paddle.base.core import VarDesc
from paddle.nn import initializer

DELTA = 0.00001


def get_uniform_min_and_max(weight):
    min_value = np.min(weight)
    max_value = np.max(weight)
    return min_value, max_value


def check_cast_op(op):
    return (
        op.type == 'cast'
        and op.attr('in_dtype') == VarDesc.VarType.FP32
        and op.attr('out_dtype') in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]
    )


class TestConstantInitializer(unittest.TestCase):
    def static_test_constant_initializer_common(
        self, init_inst, dtype="float32", value_target=0.0
    ):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=init_inst,
                )
            num_ops = 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'fill_constant')
            self.assertAlmostEqual(
                init_op.attr('value'), value_target, delta=DELTA
            )
            paddle.disable_static()
            return block

    def test_constant_initializer_default_value_static(self, dtype="float32"):
        """Test the constant initializer with default value in static graph"""
        block = self.static_test_constant_initializer_common(
            init_inst=initializer.Constant(), dtype=dtype, value_target=0.0
        )
        return block

    def test_constant_initializer_default_value_dygraph(self, dtype="float32"):
        """Test constant initializer with supplied value in dygraph"""
        with base.dygraph.guard():
            linear = nn.Linear(2, 4, weight_attr=nn.initializer.Constant())
            mat_target = np.ones((2, 4), dtype=dtype) * 0.0
            mat_linear = linear.weight.numpy()
            mismatch = np.sum(
                (mat_target - mat_linear) * (mat_target - mat_linear)
            )
            self.assertAlmostEqual(mismatch, 0.0, delta=DELTA)

    def test_constant_initializer_static(self, dtype="float32"):
        """Test constant initializer with supplied value in static graph"""
        block = self.static_test_constant_initializer_common(
            init_inst=initializer.Constant(2.3), dtype=dtype, value_target=2.3
        )
        return block

    def test_constant_initializer_dygraph(self, dtype="float32"):
        """Test constant initializer with supplied value in dygraph"""
        with base.dygraph.guard():
            linear = nn.Linear(
                2, 4, weight_attr=nn.initializer.Constant(value=2.0)
            )
            mat_target = np.ones((2, 4), dtype=dtype) * 2.0
            mat_linear = linear.weight.numpy()
            mismatch = np.sum(
                (mat_target - mat_linear) * (mat_target - mat_linear)
            )
            self.assertAlmostEqual(mismatch, 0.0, delta=DELTA)

    def test_constant_initializer_fp16(self):
        """Test constant initializer with float16"""
        block = self.test_constant_initializer_default_value_static("float16")
        block = self.test_constant_initializer_static("float16")
        self.test_constant_initializer_default_value_dygraph("float16")
        self.test_constant_initializer_dygraph("float16")

    def test_constant_initializer_bf16(self):
        """Test constant initializer with bfloat16
        No cast operator has been added here
        """
        self.test_constant_initializer_default_value_static(
            "uint16"
        )  # bfloat16
        self.test_constant_initializer_static("uint16")  # bfloat16


class TestKaimingInitializer(unittest.TestCase):
    def static_test_kaiming_initializer_common(
        self, init_inst, dtype="float32", uniform=False, is_conv=False
    ):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            shape_mat = [5, 10, 15, 20] if is_conv else [5, 10]
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=shape_mat,
                    name="param",
                    initializer=init_inst,
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            if uniform:
                self.assertEqual(init_op.type, 'uniform_random')
                if is_conv:
                    receptive_field_size = float(15 * 20)
                    limit = np.sqrt(
                        6.0 / (param.shape[1] * receptive_field_size)
                    )
                else:
                    limit = np.sqrt(6.0 / param.shape[0])
                self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
                self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            else:
                self.assertEqual(init_op.type, 'gaussian_random')
                if is_conv:
                    receptive_field_size = float(15 * 20)
                    std = np.sqrt(2.0 / (param.shape[1] * receptive_field_size))
                else:
                    std = np.sqrt(2.0 / param.shape[0])
                self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
                self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
            paddle.disable_static()

    def dygraph_test_kaiming_initializer_common(
        self, init_inst, dtype="float32", uniform=False
    ):
        linear = nn.Linear(40, 20, weight_attr=init_inst)

    def test_kaiming_dygraph(self):
        self.dygraph_test_kaiming_initializer_common(
            init_inst=initializer.KaimingUniform(),
            dtype="float32",
            uniform=True,
        )
        self.dygraph_test_kaiming_initializer_common(
            init_inst=initializer.KaimingNormal(),
            dtype="float32",
            uniform=False,
        )

    def test_kaiming_uniform_initializer_static(self):
        """Test Kaiming uniform initializer for matrix multiply."""
        self.static_test_kaiming_initializer_common(
            init_inst=initializer.KaimingUniform(),
            dtype="float32",
            uniform=True,
            is_conv=False,
        )

    def test_kaiming_uniform_initializer_conv_static(self):
        """Test Kaiming uniform initializer for convolutions."""
        self.static_test_kaiming_initializer_common(
            init_inst=initializer.KaimingUniform(),
            dtype="float32",
            uniform=True,
            is_conv=True,
        )

    def test_kaiming_normal_initializer_static(self):
        """Test Kaiming normal initializer for matrix multiply."""
        self.static_test_kaiming_initializer_common(
            init_inst=initializer.KaimingNormal(),
            dtype="float32",
            uniform=False,
            is_conv=False,
        )

    def test_kaiming_normal_initializer_conv_static(self):
        """Test Kaiming normal initializer for convolutions."""
        self.static_test_kaiming_initializer_common(
            init_inst=initializer.KaimingNormal(),
            dtype="float32",
            uniform=False,
            is_conv=True,
        )


class TestUniform(unittest.TestCase):
    def test_uniform_common(self, dtype="float32", seed=0):
        """Test the uniform initializer with default value"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            program.random_seed = seed
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.Uniform(),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            self.assertAlmostEqual(init_op.attr('min'), -1.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), 1.0, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), seed)

            paddle.disable_static()

            return block

    def test_uniform_initializer_default_value(
        self, dtype="float32", seed=0, min_value=-1.0, max_value=1.0
    ):
        """Test the uniform initializer with default value"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            program.random_seed = seed
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.Uniform(),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            self.assertAlmostEqual(init_op.attr('min'), min_value, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), max_value, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), seed)

            paddle.disable_static()

            return block

    def test_uniform_initializer(
        self, dtype="float32", seed=0, min_value=-4.2, max_value=3.1
    ):
        """Test uniform initializer with supplied attributes"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            program.random_seed = seed
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.Uniform(min_value, max_value),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            self.assertAlmostEqual(init_op.attr('min'), min_value, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), max_value, delta=DELTA)

            paddle.disable_static()

            return block

    def test_uniform_initializer_two_op(
        self, dtype="float32", seed=123, min_value=-4.2, max_value=0.0
    ):
        """Test uniform initializer with supplied attributes"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            program.random_seed = seed
            block = program.global_block()
            for i in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.Uniform(min_value, float(i)),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op0 = block.ops[0]
            self.assertEqual(init_op0.type, 'uniform_random')
            self.assertAlmostEqual(init_op0.attr('min'), min_value, delta=DELTA)
            self.assertAlmostEqual(init_op0.attr('max'), 0.0, delta=DELTA)
            self.assertEqual(init_op0.attr("seed"), seed)

            paddle.disable_static()

            return block

    def test_uniform_initializer_fp16(self):
        """Test uniform initializer with float16"""
        block = self.test_uniform_initializer_default_value("float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_uniform_initializer(dtype="float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_uniform_initializer_two_op("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_uniform_initializer_bf16(self):
        """Test uniform initializer with bfloat16"""
        block = self.test_uniform_initializer_default_value(
            "uint16"
        )  # bfloat16
        block = self.test_uniform_initializer(dtype="uint16")  # bfloat16
        block = self.test_uniform_initializer_two_op("uint16")  # bfloat16

    def test_uniform_initializer_dygraph(self):
        """Test uniform initializer in dygraph model."""
        paddle.disable_static()

        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5),
        )
        linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr)

        min_value, max_value = get_uniform_min_and_max(linear.weight.numpy())
        self.assertTrue(
            min_value >= -0.5, f'min value {min_value} should >= -0.5'
        )
        self.assertTrue(
            max_value <= 0.5, f'max value {max_value} should <= 0.5'
        )


class TestNormal(unittest.TestCase):
    def test_normal_initializer_default_value(self):
        """Test the normal initializer with default value"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.Normal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), 1.0, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

            paddle.disable_static()

    def test_normal_initializer(self, dtype="float32"):
        """Test normal initializer with supplied attributes"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.Normal(2.3, 1.9),
                )
            num_ops = 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            self.assertAlmostEqual(init_op.attr('mean'), 2.3, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), 1.9, delta=DELTA)

            paddle.disable_static()

            return block

    def test_normal_initializer_complex(self, dtype="complex64"):
        """Test normal initializer with complex dtype"""
        with paddle.pir_utils.OldIrGuard():
            with static_guard():
                program = framework.Program()
                block = program.global_block()
                for _ in range(2):
                    block.create_parameter(
                        dtype=dtype,
                        shape=[5, 10],
                        name="param",
                        initializer=initializer.Normal(2.3 + 2.3j, 1.9),
                    )
                num_ops = 1
                self.assertEqual(len(block.ops), num_ops)
                init_op = block.ops[0]
                self.assertEqual(init_op.type, 'gaussian_random')
                self.assertAlmostEqual(init_op.attr('mean'), 2.3, delta=DELTA)
                self.assertAlmostEqual(init_op.attr('std'), 1.9, delta=DELTA)

            return block

    def test_normal_initializer_fp16(self):
        """Test normal initializer with float16"""
        block = self.test_normal_initializer("float16")

    def test_normal_initializer_bf16(self):
        """Test normal initializer with bfloat16"""
        block = self.test_normal_initializer("uint16")  # bfloat16

    def test_normal_initializer_complex64(self):
        """Test normal initializer with complex64"""
        block = self.test_normal_initializer_complex("complex64")

    def test_normal_initializer_complex128(self):
        """Test normal initializer with complex128"""
        block = self.test_normal_initializer_complex("complex128")

    def test_normal_initializer_dygraph(self):
        """Test normal initializer in dygraph model."""
        paddle.disable_static()

        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
        )
        linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr)

    def test_normal_initializer_complex_error(self):
        with self.assertRaises(ValueError):
            paddle.nn.initializer.Normal(mean=0.1 + 0.2j, std=2.0)


class TestTruncatedNormal(unittest.TestCase):
    def test_truncated_normal_initializer_default_value(self):
        """Test the truncated normal initializer with default value"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.TruncatedNormal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'truncated_gaussian_random')
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), 1.0, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)
            self.assertAlmostEqual(init_op.attr('a'), -2.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('b'), 2.0, delta=DELTA)

            paddle.disable_static()

    def test_truncated_normal_initializer(self, dtype="float32"):
        """Test truncated normal initializer with supplied attributes"""
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.TruncatedNormal(2.3, 1.9),
                )
            num_ops = 2 if dtype in ["float16", "uint16"] else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'truncated_gaussian_random')
            self.assertAlmostEqual(init_op.attr('mean'), 2.3, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), 1.9, delta=DELTA)

            paddle.disable_static()

            return block

    def test_truncated_normal_initializer_fp16(self):
        """Test truncated normal initializer with float16"""
        paddle.enable_static()

        block = self.test_truncated_normal_initializer("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_truncated_normal_initializer_bf16(self):
        """Test truncated normal initializer with bfloat16"""
        paddle.enable_static()

        block = self.test_truncated_normal_initializer("uint16")  # bfloat16
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_truncated_normal_initializer_fp64(self):
        """Test truncated normal initializer with float64"""
        with static_guard():
            # Only test whether float64 data can be generated without error
            _ = self.test_truncated_normal_initializer("float64")  # float64

    def test_truncated_normal_initializer_dygraph(self):
        """Test truncated normal initializer in dygraph model."""
        paddle.disable_static()

        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=2.0
            ),
        )
        linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr)

    def test_truncated_normal_initializer_cutoff(self):
        """Test truncated normal initializer with cutoff"""
        paddle.disable_static()

        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=1.0, a=-10.0, b=10.0
            ),
        )
        linear = paddle.nn.Linear(100, 500, weight_attr=weight_attr)
        result = (linear.weight >= -10.0).all() * (linear.weight <= 10.0).all()
        np.testing.assert_equal(result.numpy(), np.array(True))

    def test_truncated_normal_initializer_cutoff2(self):
        """Test truncated normal initializer with cutoff"""
        paddle.disable_static()

        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight2",
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=1.0, a=-1.0, b=1.0
            ),
        )
        linear = paddle.nn.Linear(100, 500, weight_attr=weight_attr)
        result = (linear.weight >= -1.0).all() * (linear.weight <= 1.0).all()
        np.testing.assert_equal(result.numpy(), np.array(True))


class TestXavierUniform(unittest.TestCase):
    def test_xavier_uniform_initializer(self):
        """Test Xavier initializer with uniform distribution on
        for matrix multiply.
        """
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.XavierUniform(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            limit = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

            paddle.disable_static()

    def test_xavier_uniform_initializer_conv(self):
        """Test Xavier initializer with uniform distribution on
        for convolutions.
        """
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=initializer.XavierUniform(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            receptive_field_size = float(15 * 20)
            limit = np.sqrt(
                6.0 / ((param.shape[0] + param.shape[1]) * receptive_field_size)
            )
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_xavier_uniform_initializer_dygraph(self):
        """Test xavier uniform initializer in dygraph model."""
        paddle.disable_static()

        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.XavierUniform(),
        )
        linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr)


class TestXavierNormal(unittest.TestCase):
    def test_xavier_normal_initializer(self):
        """Test Xavier initializer with normal distribution on
        for matrix multiply.
        """
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=initializer.XavierNormal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            std = np.sqrt(2.0 / (param.shape[0] + param.shape[1]))
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

            paddle.disable_static()

    def test_xavier_normal_initializer_conv(self):
        """Test Xavier initializer with normal distribution on
        for convolutions.
        """
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=initializer.XavierNormal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            receptive_field_size = float(15 * 20)
            std = np.sqrt(
                2.0 / ((param.shape[0] + param.shape[1]) * receptive_field_size)
            )
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

            paddle.disable_static()

    def test_xavier_normal_initializer_dygraph(self):
        """Test xavier normal initializer in dygraph model."""
        paddle.disable_static()

        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.XavierNormal(),
        )
        linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr)


class TestAssign(unittest.TestCase):
    def test_assign_initializer(self, dtype="float32"):
        """Test the numpy array initializer with supplied arguments"""
        paddle.enable_static()

        import numpy

        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            np_array = numpy.random.random(10000).astype(dtype)
            for _ in range(2):
                block.create_parameter(
                    dtype=np_array.dtype,
                    shape=np_array.shape,
                    name="param",
                    initializer=initializer.Assign(np_array),
                )
            num_ops = 2 if dtype in ["float16", "uint16"] else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'assign_value')
            values = framework.extract_plain_list(init_op.attr('values'))
            assert values == np_array.ravel().tolist()
            paddle.disable_static()

            return block

    def test_assign_initializer_fp16(self):
        """Test the numpy array initializer with float16"""
        block = self.test_assign_initializer("float16")
        self.assertTrue(block.ops[1])

    def test_assign_initializer_bf16(self):
        """Test the numpy array initializer with bfloat16"""
        block = self.test_assign_initializer("uint16")  # bfloat16
        self.assertTrue(block.ops[1])

    def test_assign_initializer_dygraph_1(self):
        """Test assign initializer in dygraph model."""
        paddle.disable_static()

        weight_attr_1 = paddle.framework.ParamAttr(
            name="linear_weight_1",
            initializer=paddle.nn.initializer.Assign(np.array([2, 2])),
        )
        linear_1 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_1)

        self.assertTrue((linear_1.weight.numpy() == [2.0, 2.0]).all(), '')

    def test_assign_initializer_dygraph_2(self):
        """Test assign initializer in dygraph model."""
        paddle.disable_static()

        weight_attr_2 = paddle.framework.ParamAttr(
            name="linear_weight_2",
            initializer=paddle.nn.initializer.Assign([2, 2]),
        )
        linear_2 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_2)

        self.assertTrue((linear_2.weight.numpy() == [2.0, 2.0]).all(), '')

    def test_assign_initializer_dygraph_3(self):
        """Test assign initializer in dygraph model."""
        paddle.disable_static()

        weight_attr_3 = paddle.framework.ParamAttr(
            name="linear_weight_3",
            initializer=paddle.nn.initializer.Assign(paddle.full([2], 2)),
        )
        linear_3 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_3)

        self.assertTrue((linear_3.weight.numpy() == [2.0, 2.0]).all(), '')

    def test_assign_initializer_dygraph_4(self):
        """Test assign initializer in dygraph model."""
        paddle.disable_static()

        weight_attr_4 = paddle.framework.ParamAttr(
            name="linear_weight_4",
            initializer=paddle.nn.initializer.Assign((2, 2)),
        )
        linear_4 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_4)

        self.assertTrue((linear_4.weight.numpy() == [2.0, 2.0]).all(), '')


if __name__ == '__main__':
    unittest.main()
