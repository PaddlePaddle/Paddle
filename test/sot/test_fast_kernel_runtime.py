# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.utils.envs import ENV_SOT_ENABLE_FAST_KERNEL_CODEGEN
from paddle.utils.environments import EnvironmentVariableGuard


def fast_kernel_case(shape, x, y):
    height = shape[0]
    width = shape[1]
    z = paddle.reshape(x, [height, width])
    return [z + y]


def fast_kernel_add_case(x, y):
    return [x + y]


def fast_kernel_batch_norm_case(x, bn):
    return [bn(x)]


def assert_fast_kernel_source(test_case, source_code):
    test_case.assertNotIn("_kernel_run_phi_kernel", source_code)
    test_case.assertNotIn("run_program", source_code)


def is_cinn_kernel_source(source_code):
    return "_kernel_run_cinn_jit_kernel" in source_code


class TestFastKernelRuntime(TestCaseBase):
    def test_direct_kernel_pybind_api(self):
        from paddle.base import core

        self.assertTrue(hasattr(core.eager, "kernel_ops"))
        self.assertTrue(hasattr(core.eager.kernel_ops, "add"))
        self.assertTrue(
            hasattr(core.eager.kernel_ops, "get_phi_kernel_op_info")
        )
        x = paddle.ones([2], dtype="float32")
        y = paddle.ones([2], dtype="float32")
        out = core.eager.kernel_ops.add(x, y)
        self.assert_nest_match(out, paddle.to_tensor([2.0, 2.0]))

    def test_direct_cinn_kernel_pybind_api_without_cinn(self):
        from paddle.base import core

        self.assertTrue(hasattr(core.eager, "kernel_ops"))
        self.assertTrue(hasattr(core.eager.kernel_ops, "run_cinn_jit_kernel"))
        if not core.is_compiled_with_cinn():
            with self.assertRaisesRegex(RuntimeError, "does not fall back"):
                core.eager.kernel_ops.run_cinn_jit_kernel(None, [])

    def test_sot_fast_kernel_runtime(self):
        shape = paddle.to_tensor([2, 3], dtype="int32")
        x = paddle.arange(6, dtype="float32")
        y = paddle.ones([2, 3], dtype="float32")
        _, partial_program = paddle.jit.to_static(
            fast_kernel_case, full_graph=True
        ).get_concrete_program(shape, x, y)

        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_FAST_KERNEL_CODEGEN, True),
            paddle.no_grad(),
        ):
            out = partial_program.sot_call([shape, x, y])

        self.assert_nest_match(out, fast_kernel_case(shape, x, y))
        source_code = partial_program.fast_kernel_runtime.source_code
        if is_cinn_kernel_source(source_code):
            self.assertIn("_kernel_run_cinn_jit_kernel", source_code)
        else:
            self.assertIn("_kernel_add", source_code)
            self.assertIn("_kernel_reshape", source_code)
        self.assertNotIn("_kernel_full_int_array", source_code)
        assert_fast_kernel_source(self, source_code)

    def test_sot_fast_kernel_runtime_add(self):
        x = paddle.ones([2], dtype="float32")
        y = paddle.ones([2], dtype="float32")
        _, partial_program = paddle.jit.to_static(
            fast_kernel_add_case, full_graph=True
        ).get_concrete_program(x, y)

        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_FAST_KERNEL_CODEGEN, True),
            paddle.no_grad(),
        ):
            out = partial_program.sot_call([x, y])

        self.assert_nest_match(out, fast_kernel_add_case(x, y))
        source_code = partial_program.fast_kernel_runtime.source_code
        if is_cinn_kernel_source(source_code):
            self.assertIn("_kernel_run_cinn_jit_kernel", source_code)
        else:
            self.assertIn("_kernel_add", source_code)
        assert_fast_kernel_source(self, source_code)

    def test_sot_fast_kernel_runtime_batch_norm_eval(self):
        x = paddle.rand([1, 3, 8, 8])
        bn = paddle.nn.BatchNorm2D(3)
        bn.eval()
        _, partial_program = paddle.jit.to_static(
            fast_kernel_batch_norm_case, full_graph=True
        ).get_concrete_program(x, bn)

        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_FAST_KERNEL_CODEGEN, True),
            paddle.no_grad(),
        ):
            out = partial_program.sot_call([x])

        expected = fast_kernel_batch_norm_case(x, bn)
        self.assert_nest_match(out, expected)
        self.assertEqual(out[0].shape, expected[0].shape)
        source_code = partial_program.fast_kernel_runtime.source_code
        if is_cinn_kernel_source(source_code):
            self.assertIn("_kernel_run_cinn_jit_kernel", source_code)
        else:
            self.assertIn("_kernel_batch_norm", source_code)
            self.assertIn("return (batch_norm", source_code)
            self.assertNotIn("_kernel_batch_norm_", source_code)
        assert_fast_kernel_source(self, source_code)

    def test_sot_fast_kernel_runtime_no_run_program_fallback(self):
        shape = paddle.to_tensor([2, 3], dtype="int32")
        x = paddle.arange(6, dtype="float32")
        y = paddle.ones([2, 3], dtype="float32")
        _, partial_program = paddle.jit.to_static(
            fast_kernel_case, full_graph=True
        ).get_concrete_program(shape, x, y)

        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_FAST_KERNEL_CODEGEN, True),
            self.assertRaisesRegex(RuntimeError, "does not fall back"),
        ):
            partial_program.sot_call([shape, x, y])


if __name__ == "__main__":
    unittest.main()
