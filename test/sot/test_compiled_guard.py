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

from __future__ import annotations

import unittest

import paddle
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.opcode_translator.executor.executor_cache import (
    OpcodeExecutorCache,
)
from paddle.jit.sot.opcode_translator.executor.guard import make_compiled_guard
from paddle.jit.sot.utils import (
    ENV_SOT_ENABLE_COMPILED_GUARD,
    ENV_SOT_ENABLE_COMPILED_GUARD_TREE,
    ENV_SOT_ENABLE_STRICT_GUARD_CHECK,
    ENV_SOT_UNSAFE_CACHE_FASTPATH,
)
from paddle.utils.environments import EnvironmentVariableGuard


def compiled_guard_helper(x):
    return x + 1


class TestCompiledGuard(unittest.TestCase):
    def setUp(self):
        OpcodeExecutorCache().clear()

    def tearDown(self):
        OpcodeExecutorCache().clear()

    def get_only_guard(self):
        cache = OpcodeExecutorCache().cache
        self.assertEqual(len(cache), 1)
        guarded_fns, _ = next(iter(cache.values()))
        self.assertEqual(len(guarded_fns), 1)
        return guarded_fns[0][1]

    def test_compiled_guard_hit_and_shape_miss(self):
        def fn(x):
            return x + 1

        compiled_fn = symbolic_translate(fn)
        with EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True):
            compiled_fn(paddle.ones([2, 3]))
            guard_fn = self.get_only_guard()
            self.assertTrue(hasattr(guard_fn, "compiled_guard"))
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            compiled_fn(paddle.ones([2, 3]))
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            compiled_fn(paddle.ones([3, 2]))
            self.assertEqual(OpcodeExecutorCache().translate_count, 2)

    def test_compiled_guard_keeps_python_type_guard(self):
        def fn(x, flag):
            if flag:
                return x + 1
            return x - 1

        compiled_fn = symbolic_translate(fn)
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            compiled_fn(paddle.ones([2, 3]), 1)
            guard_fn = self.get_only_guard()
            self.assertTrue(hasattr(guard_fn, "compiled_guard"))
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            compiled_fn(paddle.ones([2, 3]), True)
            self.assertEqual(OpcodeExecutorCache().translate_count, 2)

    def test_compiled_guard_nested_container_hit_and_value_miss(self):
        def fn(x, cfg):
            if cfg["enabled"]:
                return x + cfg["biases"][0]
            return x - cfg["biases"][1]

        compiled_fn = symbolic_translate(fn)
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            compiled_fn(
                paddle.ones([2, 3]), {"enabled": True, "biases": [1, 2]}
            )
            guard_fn = self.get_only_guard()
            self.assertTrue(hasattr(guard_fn, "compiled_guard"))
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            compiled_fn(
                paddle.ones([2, 3]), {"enabled": True, "biases": [1, 2]}
            )
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            compiled_fn(
                paddle.ones([2, 3]), {"enabled": True, "biases": [2, 2]}
            )
            self.assertEqual(OpcodeExecutorCache().translate_count, 2)

    def test_compiled_guard_grad_enabled_miss(self):
        def fn(x):
            return x + 1

        compiled_fn = symbolic_translate(fn)
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            compiled_fn(paddle.ones([2, 3]))
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            with paddle.no_grad():
                compiled_fn(paddle.ones([2, 3]))
            self.assertEqual(OpcodeExecutorCache().translate_count, 2)

    def test_compiled_guard_global_function_weakref_miss(self):
        def fn(x):
            return compiled_guard_helper(x)

        compiled_fn = symbolic_translate(fn)
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            global compiled_guard_helper

            original_helper = compiled_guard_helper
            try:
                compiled_guard_helper = lambda x: x + 1
                compiled_fn(paddle.ones([2, 3]))
                guard_fn = self.get_only_guard()
                self.assertTrue(hasattr(guard_fn, "compiled_guard"))
                self.assertEqual(OpcodeExecutorCache().translate_count, 1)

                compiled_guard_helper = lambda x: x - 1
                compiled_fn(paddle.ones([2, 3]))
                self.assertEqual(OpcodeExecutorCache().translate_count, 2)
            finally:
                compiled_guard_helper = original_helper

    def test_compiled_guard_layer_hook_miss(self):
        class SimpleLayer(paddle.nn.Layer):
            def forward(self, x):
                return x + 1

        def fn(x, layer):
            return layer(x)

        compiled_fn = symbolic_translate(fn)
        layer = SimpleLayer()
        layer.eval()
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            compiled_fn(paddle.ones([2, 3]), layer)
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            hook = layer.register_forward_pre_hook(lambda _layer, inputs: None)
            try:
                compiled_fn(paddle.ones([2, 3]), layer)
                self.assertGreater(OpcodeExecutorCache().translate_count, 1)
            finally:
                hook.remove()

    def test_compiled_guard_layer_forward_override_miss(self):
        class SimpleLayer(paddle.nn.Layer):
            def forward(self, x):
                return x + 1

        def fn(x, layer):
            return layer(x)

        compiled_fn = symbolic_translate(fn)
        layer = SimpleLayer()
        layer.eval()
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            compiled_fn(paddle.ones([2, 3]), layer)
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            layer.forward = lambda x: x - 1
            compiled_fn(paddle.ones([2, 3]), layer)
            self.assertEqual(OpcodeExecutorCache().translate_count, 2)

    def test_compiled_guard_layer_unrelated_attr_still_hits(self):
        class SimpleLayer(paddle.nn.Layer):
            def forward(self, x):
                return x + 1

        def fn(x, layer):
            return layer(x)

        compiled_fn = symbolic_translate(fn)
        layer = SimpleLayer()
        layer.eval()
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            compiled_fn(paddle.ones([2, 3]), layer)
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            layer.unused_for_guard = object()
            compiled_fn(paddle.ones([2, 3]), layer)
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

    def test_compiled_guard_constructor_error_is_not_suppressed(self):
        def python_guard(_frame):
            return True

        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            self.assertRaises(ValueError),
        ):
            make_compiled_guard(
                [("unknown_guard_kind", (("local", "x"),), 1)],
                python_guard,
            )

    def test_compiled_guard_tree_skips_earlier_linear_guards(self):
        def fn(x, mode):
            if mode == 0:
                return x + 1
            if mode == 1:
                return x + 2
            if mode == 2:
                return x + 3
            return x + 4

        compiled_fn = symbolic_translate(fn)
        modes = [0, 1, 2, 3]

        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD_TREE, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, False),
            EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, False),
        ):
            x = paddle.ones([2, 3])
            for mode in modes:
                compiled_fn(x, mode)

            cache = OpcodeExecutorCache().cache
            self.assertEqual(len(cache), 1)
            guarded_fns, _ = next(iter(cache.values()))
            self.assertEqual(len(guarded_fns), len(modes))
            self.assertEqual(OpcodeExecutorCache().translate_count, len(modes))

            for index in range(3):
                custom_code, _guard_fn = guarded_fns[index]

                def raising_guard(_frame, index=index):
                    raise AssertionError(f"linear guard {index} should not run")

                raising_guard.expr = "compiled guard tree test guard"
                raising_guard.inlined_expr = raising_guard.expr
                guarded_fns[index] = (custom_code, raising_guard)

            out = compiled_fn(x, modes[-1])
            self.assertEqual(list(out.shape), [2, 3])
            self.assertEqual(OpcodeExecutorCache().translate_count, len(modes))

    def test_compiled_guard_distributed_tensor_meta(self):
        if not paddle.is_compiled_with_distribute():
            return

        import paddle.distributed as dist

        def fn(x, y):
            return x + y

        x = paddle.ones([2, 2])
        x.stop_gradient = False
        y = paddle.zeros([2, 2])
        y.stop_gradient = False
        mesh1 = dist.ProcessMesh([0, 1], dim_names=["x"])
        mesh2 = dist.ProcessMesh([0, 1], dim_names=["y"])
        mesh3 = dist.ProcessMesh([0, 2], dim_names=["x"])
        dist_x1 = dist.shard_tensor(
            x, mesh1, [dist.Replicate()], stop_gradient=False
        )
        dist_y1 = dist.shard_tensor(
            y, mesh1, [dist.Replicate()], stop_gradient=False
        )
        dist_x2 = dist.shard_tensor(x, mesh2, [dist.Replicate()])
        dist_y2 = dist.shard_tensor(y, mesh2, [dist.Replicate()])
        dist_x3 = dist.shard_tensor(x, mesh3, [dist.Replicate()])
        dist_y3 = dist.shard_tensor(y, mesh3, [dist.Replicate()])

        compiled_fn = symbolic_translate(fn)
        with (
            EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
            EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, True),
        ):
            compiled_fn(dist_x1, dist_y1)
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            compiled_fn(dist_x2, dist_y2)
            self.assertEqual(OpcodeExecutorCache().translate_count, 1)

            compiled_fn(dist_x3, dist_y3)
            self.assertEqual(OpcodeExecutorCache().translate_count, 2)


if __name__ == "__main__":
    unittest.main()
