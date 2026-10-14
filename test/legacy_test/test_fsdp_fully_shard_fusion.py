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

"""Single-process unit tests for ``paddle.distributed.fsdp.fully_shard_fusion``.

The module is loaded directly from the source tree (not from an installed
wheel, which may lag behind) so the tests always exercise the checked-out
code. The three behaviours covered here are pure/host-side logic that needs
no distributed launch:

* ``_find_exposed_param_names`` -- which of a layer's own parameters leak into
  its outputs, including the bounded-depth walk.
* the fused-kernel fp32 gradient hand-off inside the grad hook.
* the tuple branch of the forward post hook that splices exposed parameters
  back untouched while routing the rest through the backward all_gather hook.
"""

import importlib.util
import os
import sys
import types
import unittest

import numpy as np

import paddle


def _load_module_under_test():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(
        os.path.join(
            here, "..", "..", "python", "paddle", "distributed", "fsdp",
            "fully_shard_fusion.py",
        )
    )
    spec = importlib.util.spec_from_file_location("fsf_under_test", src)
    module = importlib.util.module_from_spec(spec)
    # dataclass() looks the module up in sys.modules by __module__, so register
    # it before executing the body.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fsf = _load_module_under_test()


class _NoopCommManager:
    """comm_manager stand-in: the hooks only need these to be callable.

    ``buffer_manager.tie_param_name`` is read when registering the tie-weight
    pre hook; a value no real param name matches keeps that path dormant.
    """

    def __init__(self):
        self.buffer_manager = types.SimpleNamespace(tie_param_name=None)

    def _maybe_zero_grads(self):
        pass

    def _ensure_grads_writable(self, param):
        pass

    def shard_params(self, params, is_backward=False):
        pass

    def reduce_scatter_grads(self, param):
        pass


class _FakeParam:
    """Minimal fused parameter: a lazily sized ``main_grad`` plus a slot that
    captures the hook ``_register_grad_hook_once`` installs."""

    def __init__(self):
        self.main_grad = None
        self.registered_hook = None

    def get_main_grad(self, shape):
        if self.main_grad is None:
            self.main_grad = paddle.zeros(shape, dtype="float32")
        return self.main_grad

    def _register_grad_hook(self, hook):
        self.registered_hook = hook


class _NamedParam:
    def __init__(self, name):
        self.name = name


class _FakeLayer:
    def __init__(self, params):
        self._params = list(params)
        self.post_hook = None
        self.pre_hook = None

    def parameters(self, include_sublayers=True):
        return self._params

    def register_forward_post_hook(self, hook):
        self.post_hook = hook

    def register_forward_pre_hook(self, hook):
        self.pre_hook = hook


# PLACEHOLDER_TESTS


class TestFindExposedParamNames(unittest.TestCase):
    def _tensor(self):
        return paddle.to_tensor([1.0, 2.0])

    def test_empty_own_names_short_circuits(self):
        t = self._tensor()
        self.assertEqual(fsf._find_exposed_param_names(t, set()), set())

    def test_bare_tensor_hit_and_miss(self):
        t = self._tensor()
        self.assertEqual(fsf._find_exposed_param_names(t, {t.name}), {t.name})
        self.assertEqual(fsf._find_exposed_param_names(t, {"not-a-name"}), set())

    def test_tuple_returns_only_exposed_members(self):
        hidden, weight, bias = self._tensor(), self._tensor(), self._tensor()
        found = fsf._find_exposed_param_names(
            (hidden, weight, bias), {weight.name, bias.name}
        )
        self.assertEqual(found, {weight.name, bias.name})

    def test_dict_values_are_scanned(self):
        hidden, weight = self._tensor(), self._tensor()
        found = fsf._find_exposed_param_names(
            {"h": hidden, "w": weight}, {weight.name}
        )
        self.assertEqual(found, {weight.name})

    def test_non_container_returns_empty(self):
        self.assertEqual(fsf._find_exposed_param_names(123, {"x"}), set())

    def test_depth_bound_stops_deep_nesting(self):
        # A tensor two container levels deep is still found (the tensor check
        # precedes the guard); one level deeper trips ``_depth >= 2``.
        t = self._tensor()
        self.assertEqual(
            fsf._find_exposed_param_names(((t,),), {t.name}), {t.name}
        )
        self.assertEqual(
            fsf._find_exposed_param_names((((t,),),), {t.name}), set()
        )


# PLACEHOLDER_TESTS2


class TestFusedKernelGradHook(unittest.TestCase):
    """Covers the fp32 hand-off in the grad hook from ``_register_grad_hook_once``."""

    def _make_hooked_param(self):
        fsdp = object.__new__(fsf.FullyShardFusion)
        fsdp.comm_manager = _NoopCommManager()
        param = _FakeParam()
        fsdp._register_grad_hook_once(param)
        self.assertIsNotNone(param.registered_hook)
        return param

    def test_single_fp32_grad_wins_over_autograd_grad(self):
        param = self._make_hooked_param()
        param._fused_kernel_fp32_grad = paddle.to_tensor([1.0, 2.0, 3.0])
        # The autograd-delivered (bf16-rounded) grad must be ignored and freed.
        incoming = paddle.to_tensor([9.0, 9.0, 9.0])
        param.registered_hook(incoming)
        np.testing.assert_array_equal(param.main_grad.numpy(), [1.0, 2.0, 3.0])
        self.assertIsNone(param._fused_kernel_fp32_grad)
        self.assertFalse(incoming._is_initialized())

    def test_fp32_grad_list_added_in_order(self):
        param = self._make_hooked_param()
        param._fused_kernel_fp32_grad = [
            paddle.to_tensor([1.0, 1.0]),
            paddle.to_tensor([10.0, 20.0]),
        ]
        param.registered_hook(None)
        np.testing.assert_array_equal(param.main_grad.numpy(), [11.0, 21.0])
        self.assertIsNone(param._fused_kernel_fp32_grad)

    def test_without_fp32_grad_falls_back_to_autograd_grad(self):
        param = self._make_hooked_param()
        incoming = paddle.to_tensor([2.0, 4.0])
        param.registered_hook(incoming)
        np.testing.assert_array_equal(param.main_grad.numpy(), [2.0, 4.0])
        self.assertFalse(incoming._is_initialized())


# PLACEHOLDER_TESTS3


class TestForwardPostHookTupleBranch(unittest.TestCase):
    """Covers the tuple branch of the forward post hook built by
    ``_register_fusion_layer_hooks``."""

    def _capture_post_hook(self, param_names):
        fsdp = object.__new__(fsf.FullyShardFusion)
        fsdp.comm_manager = _NoopCommManager()
        layer = _FakeLayer([_NamedParam(n) for n in param_names])
        fsdp._register_fusion_layer_hooks(layer, recursive=False)
        self.assertIsNotNone(layer.post_hook)
        return layer

    def test_exposed_param_spliced_back_untouched(self):
        hidden = paddle.to_tensor([1.0, 2.0])
        weight = paddle.to_tensor([3.0, 4.0])  # exposed LM-head weight
        layer = self._capture_post_hook([weight.name])
        result = layer.post_hook(layer, (), (hidden, weight))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        # Exposed param returned by identity; it must be the very same object.
        self.assertIs(result[1], weight)
        # The activation is routed through the (identity) backward hook.
        np.testing.assert_array_equal(result[0].numpy(), hidden.numpy())

    def test_all_exposed_returns_outputs_unchanged(self):
        weight = paddle.to_tensor([3.0, 4.0])
        layer = self._capture_post_hook([weight.name])
        outputs = (weight,)
        result = layer.post_hook(layer, (), outputs)
        self.assertIs(result, outputs)

    def test_no_exposed_params_routes_all_through_hook(self):
        a = paddle.to_tensor([1.0])
        b = paddle.to_tensor([2.0])
        layer = self._capture_post_hook([])  # nothing exposed
        result = layer.post_hook(layer, (), (a, b))
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0].numpy(), a.numpy())
        np.testing.assert_array_equal(result[1].numpy(), b.numpy())


if __name__ == "__main__":
    unittest.main()
