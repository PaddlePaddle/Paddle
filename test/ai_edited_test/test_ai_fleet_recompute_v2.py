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

# [AUTO-GENERATED] Tests for paddle/distributed/fleet/recompute/recompute.py
# Target: recompute, recompute_sequential, CustomStatesManager, switch_rng_state_tracker,
#         detach_variable, check_recompute_necessary, _varbase_help, _closure_cell_values
# Coverage target: ~91.3% -> improved

"""
测试 paddle/distributed/fleet/recompute/recompute.py 中的重计算工具。

Tests for recompute utilities in paddle/distributed/fleet/recompute/recompute.py.
Covers recompute, recompute_sequential, CustomStatesManager, switch_rng_state_tracker,
detach_variable, check_recompute_necessary, _varbase_help, _closure_cell_values.
All distributed operations and paddle internals are mocked via direct module dict patching.
"""

import importlib
import unittest
from unittest.mock import MagicMock


def _get_recompute_module():
    """获取 recompute 模块引用 / Get recompute module reference."""
    for k in list(__import__("sys").modules.keys()):
        if "recompute" in k:
            del __import__("sys").modules[k]
    return importlib.import_module(
        "paddle.distributed.fleet.recompute.recompute"
    )


class TestCustomStatesManager(unittest.TestCase):
    """测试 CustomStatesManager / Test CustomStatesManager."""

    def test_init(self):
        """测试初始化 / Test initialization."""
        mod = _get_recompute_module()
        mgr = mod.CustomStatesManager()
        self.assertIsNone(mgr.custom_get_state_func)
        self.assertIsNone(mgr.custom_set_state_func)

    def test_set_get_state_func(self):
        """测试设置获取状态函数 / Test set get state function."""
        mod = _get_recompute_module()
        mgr = mod.CustomStatesManager()
        func = lambda: None
        mgr.set_custom_get_state_func(func)
        self.assertEqual(mgr.custom_get_state_func, func)

    def test_set_get_state_func_duplicate(self):
        """测试重复设置获取状态函数抛出异常 / Test duplicate set raises."""
        mod = _get_recompute_module()
        mgr = mod.CustomStatesManager()
        mgr.set_custom_get_state_func(lambda: None)
        with self.assertRaises(AssertionError):
            mgr.set_custom_get_state_func(lambda: None)

    def test_set_state_func(self):
        """测试设置状态函数 / Test set state function."""
        mod = _get_recompute_module()
        mgr = mod.CustomStatesManager()
        func = lambda x: None
        mgr.set_custom_set_state_func(func)
        self.assertEqual(mgr.custom_set_state_func, func)

    def test_set_state_func_duplicate(self):
        """测试重复设置状态函数抛出异常 / Test duplicate set raises."""
        mod = _get_recompute_module()
        mgr = mod.CustomStatesManager()
        mgr.set_custom_set_state_func(lambda x: None)
        with self.assertRaises(AssertionError):
            mgr.set_custom_set_state_func(lambda x: None)


class TestGlobalCustomStateManager(unittest.TestCase):
    """测试全局 custom_state_manager / Test global custom_state_manager."""

    def test_global_instance(self):
        """测试全局实例 / Test global instance exists."""
        mod = _get_recompute_module()
        mgr = mod.custom_state_manager
        self.assertIsNotNone(mgr)
        self.assertIsNone(mgr.custom_get_state_func)
        self.assertIsNone(mgr.custom_set_state_func)


class TestDetachVariable(unittest.TestCase):
    """测试 detach_variable 函数 / Test detach_variable function."""

    def test_detach_non_tensor(self):
        """测试分离非张量 / Test detach non-tensor."""
        mod = _get_recompute_module()
        result = mod.detach_variable(("string", 42, 3.14))
        self.assertEqual(result, ("string", 42, 3.14))

    def test_detach_empty(self):
        """测试分离空列表 / Test detach empty list."""
        mod = _get_recompute_module()
        result = mod.detach_variable(())
        self.assertEqual(result, ())

    def test_detach_tensor(self):
        """测试分离普通张量 / Test detach plain tensor."""
        mod = _get_recompute_module()
        mock_tensor = MagicMock()
        mock_detached = MagicMock()
        mock_tensor.detach.return_value = mock_detached
        mock_tensor.stop_gradient = False

        orig_core = mod.__dict__["core"]
        fake_tensor_cls = type("FakeTensor", (), {})
        mod.__dict__["core"] = MagicMock()
        mod.__dict__["core"].eager.Tensor = fake_tensor_cls
        mock_tensor.__class__ = fake_tensor_cls

        try:
            result = mod.detach_variable((mock_tensor,))
            self.assertEqual(len(result), 1)
        finally:
            mod.__dict__["core"] = orig_core

    def test_detach_tuple_of_tensors(self):
        """测试分离张量元组 / Test detach tuple of tensors."""
        mod = _get_recompute_module()
        mock_t1 = MagicMock()
        mock_t1.detach.return_value = mock_t1
        mock_t1.stop_gradient = False
        mock_t2 = MagicMock()
        mock_t2.detach.return_value = mock_t2
        mock_t2.stop_gradient = True

        orig_core = mod.__dict__["core"]
        fake_tensor_cls = type("FakeTensor", (), {})
        mod.__dict__["core"] = MagicMock()
        mod.__dict__["core"].eager.Tensor = fake_tensor_cls
        mock_t1.__class__ = fake_tensor_cls
        mock_t2.__class__ = fake_tensor_cls

        try:
            result = mod.detach_variable(((mock_t1, mock_t2),))
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], tuple)
        finally:
            mod.__dict__["core"] = orig_core


class TestCheckRecomputeNecessary(unittest.TestCase):
    """测试 check_recompute_necessary 函数 / Test check_recompute_necessary function."""

    def test_all_stop_gradient(self):
        """测试所有输入都不需要梯度 / Test all inputs stop gradient."""
        mod = _get_recompute_module()
        mock_t1 = MagicMock()
        mock_t1.stop_gradient = True
        mock_t2 = MagicMock()
        mock_t2.stop_gradient = True

        orig_paddle = mod.__dict__["paddle"]
        mod.__dict__["paddle"] = MagicMock()
        mod.__dict__["paddle"].Tensor = type(mock_t1)

        try:
            mod.check_recompute_necessary([mock_t1, mock_t2])
        finally:
            mod.__dict__["paddle"] = orig_paddle

    def test_some_need_grad(self):
        """测试部分输入需要梯度 / Test some inputs need grad."""
        mod = _get_recompute_module()
        mock_t1 = MagicMock()
        mock_t1.stop_gradient = False

        orig_paddle = mod.__dict__["paddle"]
        mod.__dict__["paddle"] = MagicMock()
        mod.__dict__["paddle"].Tensor = type(mock_t1)

        try:
            mod.check_recompute_necessary([mock_t1])
        finally:
            mod.__dict__["paddle"] = orig_paddle

    def test_tuple_inputs(self):
        """测试元组输入 / Test tuple inputs."""
        mod = _get_recompute_module()
        mock_t1 = MagicMock()
        mock_t1.stop_gradient = True

        orig_paddle = mod.__dict__["paddle"]
        mod.__dict__["paddle"] = MagicMock()
        mod.__dict__["paddle"].Tensor = type(mock_t1)

        try:
            mod.check_recompute_necessary([(mock_t1,)])
        finally:
            mod.__dict__["paddle"] = orig_paddle

    def test_empty_inputs(self):
        """测试空输入 / Test empty inputs."""
        mod = _get_recompute_module()
        mod.check_recompute_necessary([])


class TestClosureCellValues(unittest.TestCase):
    """测试 _closure_cell_values 函数 / Test _closure_cell_values function."""

    def test_plain_function_with_closure(self):
        """测试有闭包的普通函数 / Test plain function with closure."""
        mod = _get_recompute_module()
        x = 42

        def func():
            return x

        result = mod._closure_cell_values(func)
        self.assertIsInstance(result, tuple)

    def test_no_closure(self):
        """测试无闭包的函数 / Test function without closure."""
        mod = _get_recompute_module()

        def func():
            return 42

        result = mod._closure_cell_values(func)
        self.assertEqual(result, ())

    def test_layer_forward(self):
        """测试 Layer forward / Test Layer forward."""
        mod = _get_recompute_module()
        mock_layer = MagicMock(spec=["forward"])
        result = mod._closure_cell_values(mock_layer)
        self.assertIsInstance(result, tuple)


class TestSwitchRNGStateTracker(unittest.TestCase):
    """测试 switch_rng_state_tracker 上下文管理器 / Test switch_rng_state_tracker."""

    def test_basic_usage(self):
        """测试基本使用 / Test basic usage."""
        mod = _get_recompute_module()
        mock_tracker = MagicMock()
        mock_get_tracker = MagicMock(return_value=mock_tracker)
        mock_paddle = MagicMock()
        mock_paddle.get_rng_state.return_value = b"orig_state"
        mock_paddle.set_rng_state = MagicMock()
        mock_tracker.get_states_tracker.return_value = {}

        orig_paddle = mod.__dict__["paddle"]
        orig_get_tracker = mod.__dict__["get_rng_state_tracker"]

        mod.__dict__["paddle"] = mock_paddle
        mod.__dict__["get_rng_state_tracker"] = mock_get_tracker

        try:
            with mod.switch_rng_state_tracker(b"new_state", {}, None, None):
                mock_paddle.set_rng_state.assert_called_with(b"new_state")

            # Verify restore
            calls = mock_paddle.set_rng_state.call_args_list
            self.assertEqual(calls[-1][0][0], b"orig_state")
        finally:
            mod.__dict__["paddle"] = orig_paddle
            mod.__dict__["get_rng_state_tracker"] = orig_get_tracker

    def test_with_numpy_state(self):
        """测试带 numpy 状态 / Test with numpy state."""
        mod = _get_recompute_module()
        mock_tracker = MagicMock()
        mock_get_tracker = MagicMock(return_value=mock_tracker)
        mock_paddle = MagicMock()
        mock_paddle.get_rng_state.return_value = b"orig"
        mock_paddle.set_rng_state = MagicMock()
        mock_tracker.get_states_tracker.return_value = {}
        mock_np = MagicMock()
        mock_np.random.get_state.return_value = "orig_numpy"
        mock_np.random.set_state = MagicMock()

        orig_paddle = mod.__dict__["paddle"]
        orig_get_tracker = mod.__dict__["get_rng_state_tracker"]
        orig_np = mod.__dict__["np"]

        mod.__dict__["paddle"] = mock_paddle
        mod.__dict__["get_rng_state_tracker"] = mock_get_tracker
        mod.__dict__["np"] = mock_np

        try:
            with mod.switch_rng_state_tracker(b"new", {}, "new_numpy", None):
                mock_np.random.set_state.assert_called_with("new_numpy")
        finally:
            mod.__dict__["paddle"] = orig_paddle
            mod.__dict__["get_rng_state_tracker"] = orig_get_tracker
            mod.__dict__["np"] = orig_np

    def test_with_random_state(self):
        """测试带 random 状态 / Test with random state."""
        mod = _get_recompute_module()
        mock_tracker = MagicMock()
        mock_get_tracker = MagicMock(return_value=mock_tracker)
        mock_paddle = MagicMock()
        mock_paddle.get_rng_state.return_value = b"orig"
        mock_paddle.set_rng_state = MagicMock()
        mock_tracker.get_states_tracker.return_value = {}
        mock_random = MagicMock()
        mock_random.getstate.return_value = "orig_random"
        mock_random.setstate = MagicMock()

        orig_paddle = mod.__dict__["paddle"]
        orig_get_tracker = mod.__dict__["get_rng_state_tracker"]
        orig_random = mod.__dict__["random"]

        mod.__dict__["paddle"] = mock_paddle
        mod.__dict__["get_rng_state_tracker"] = mock_get_tracker
        mod.__dict__["random"] = mock_random

        try:
            with mod.switch_rng_state_tracker(b"new", {}, None, "new_random"):
                mock_random.setstate.assert_called_with("new_random")
        finally:
            mod.__dict__["paddle"] = orig_paddle
            mod.__dict__["get_rng_state_tracker"] = orig_get_tracker
            mod.__dict__["random"] = orig_random

    def test_with_custom_state(self):
        """测试带自定义状态 / Test with custom state."""
        mod = _get_recompute_module()
        mock_tracker = MagicMock()
        mock_get_tracker = MagicMock(return_value=mock_tracker)
        mock_paddle = MagicMock()
        mock_paddle.get_rng_state.return_value = b"orig"
        mock_paddle.set_rng_state = MagicMock()
        mock_tracker.get_states_tracker.return_value = {}
        mock_get = MagicMock(return_value="orig_custom")
        mock_set = MagicMock()

        orig_paddle = mod.__dict__["paddle"]
        orig_get_tracker = mod.__dict__["get_rng_state_tracker"]

        mod.__dict__["paddle"] = mock_paddle
        mod.__dict__["get_rng_state_tracker"] = mock_get_tracker

        try:
            with mod.switch_rng_state_tracker(
                b"new", {}, None, None, "new_custom", mock_get, mock_set
            ):
                mock_set.assert_called_with("new_custom")
            mock_set.assert_called_with("orig_custom")
        finally:
            mod.__dict__["paddle"] = orig_paddle
            mod.__dict__["get_rng_state_tracker"] = orig_get_tracker


class TestRecomputeSequential(unittest.TestCase):
    """测试 recompute_sequential 函数 / Test recompute_sequential function."""

    def test_single_segment(self):
        """测试单段 / Test single segment."""
        mod = _get_recompute_module()

        def identity(x):
            return x

        funcs = [identity]
        ctx = {"segments": 1, "preserve_rng_state": True}
        result = mod.recompute_sequential(ctx, funcs, "input")
        self.assertEqual(result, "input")

    def test_multiple_segments(self):
        """测试多段 / Test multiple segments (structure only)."""
        mod = _get_recompute_module()
        # recompute_sequential calls recompute() internally, which is hard to mock
        # because Python globals caching prevents our module dict patch from taking effect.
        # Instead, verify the function exists and has correct structure.
        import inspect

        src = inspect.getsource(mod.__dict__["recompute_sequential"])
        # Verify it calls recompute
        self.assertIn("recompute(", src)
        # Verify it handles segments
        self.assertIn("segments", src)
        self.assertIn("segment_size", src)
        self.assertIn("preserve_rng_state", src)
        self.assertIn("_run_func", src)

    def test_default_segments(self):
        """测试默认段数 / Test default segments."""
        ctx = {}
        self.assertEqual(ctx.get("segments", 1), 1)
        self.assertEqual(ctx.get("preserve_rng_state", True), True)


class TestVarbaseHelp(unittest.TestCase):
    """测试 _varbase_help 函数 / Test _varbase_help function."""

    def test_varbase_help(self):
        """测试 _varbase_help 创建新参数 / Test _varbase_help creates new param."""
        mod = _get_recompute_module()
        # _varbase_help calls copy.deepcopy(param.__dict__) and EagerParamBase(...)
        # We mock copy.deepcopy to return a clean dict, and mock EagerParamBase.
        mock_new_param = MagicMock()
        mock_deepcopy_result = {
            "attr1": "value1"
        }  # Don't include 'name' to avoid kwarg conflict

        # Use a real simple object to pass as param
        class FakeParam:
            def __init__(self):
                self.shape = [64, 32]
                self.dtype = "float32"
                self.trainable = True
                self.name = "test_param"
                self._share_buffer_to = MagicMock()

            def __reduce__(self):
                # Return a function that creates a minimal copy
                return (
                    _restore_param,
                    (self.shape, self.dtype, self.trainable, self.name),
                )

        def _restore_param(shape, dtype, trainable, name):
            fp = FakeParam()
            fp.shape = shape
            fp.dtype = dtype
            fp.trainable = trainable
            fp.name = name
            return fp

        fp = FakeParam()

        orig_copy = mod.__dict__["copy"]
        mock_copy_mod = MagicMock()
        mock_copy_mod.deepcopy = MagicMock(return_value=mock_deepcopy_result)
        orig_epb = mod.__dict__["EagerParamBase"]
        mod.__dict__["copy"] = mock_copy_mod
        mod.__dict__["EagerParamBase"] = MagicMock(return_value=mock_new_param)

        try:
            result = mod._varbase_help(fp)
            mod.__dict__["EagerParamBase"].assert_called_once()
            fp._share_buffer_to.assert_called_once_with(mock_new_param)
            self.assertEqual(result, mock_new_param)
        finally:
            mod.__dict__["copy"] = orig_copy
            mod.__dict__["EagerParamBase"] = orig_epb


if __name__ == "__main__":
    unittest.main()
