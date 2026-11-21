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

"""Regression test for SOT ``KeyError: 'self'`` when a compiled method calls
another compiled method via ``self.xxx()``.

Paddle Issue: https://github.com/PaddlePaddle/Paddle/issues/79325
Paddle PR: https://github.com/PaddlePaddle/Paddle/pull/79326

When both ``forward`` and ``step`` are wrapped with
``paddle.jit.to_static``, and ``step`` calls ``self.forward(x)``, SOT's
inline executor previously lost the ``self`` binding, causing
``KeyError: 'self'``.

Fix: ``UserDefinedFunctionVariable.from_value`` detects bound
``StaticFunction`` (``class_instance is not None``) and returns a
``MethodVariable``, so that ``load_method`` preserves the ``self``
argument in ``CALL_METHOD`` dispatch.
"""

import unittest

from test_case_base import TestCaseBase

import paddle


class ReproLayer(paddle.nn.Layer):
    """Minimal reproduction: 1 sub-module, 2 compiled methods,
    one calls the other."""

    def __init__(self):
        super().__init__()
        self.dim = 8
        self.conv = paddle.nn.Conv1D(8, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(paddle.nn.functional.relu(x[:, : self.dim]))

    def step(self, x):
        out = self.forward(x)
        return out.mean()


class TestBoundStaticMethodCall(TestCaseBase):
    """SOT must correctly handle ``self.method()`` when both caller
    and callee are compiled with ``paddle.jit.to_static``."""

    def _generate_input(self):
        return paddle.randn([2, 8, 32])

    def test_no_to_static(self):
        """Sanity: dynamic execution must work."""
        m = ReproLayer()
        x = self._generate_input()
        out = m.step(x)
        self.assertIsNotNone(out)

    def test_forward_only_compiled(self):
        """Only forward compiled — should work (1 compiled method)."""
        m = ReproLayer()
        m.forward = paddle.jit.to_static(m.forward)
        x = self._generate_input()
        # step calls self.forward dynamically
        out = m.step(x)
        self.assertIsNotNone(out)

    def test_both_compiled_no_assertion(self):
        """Both methods compiled, assert no crash (KeyError: self)."""
        m = ReproLayer()
        m.forward = paddle.jit.to_static(m.forward)
        m.step = paddle.jit.to_static(m.step)
        x = self._generate_input()
        try:
            out = m.step(x)
            self.assertIsNotNone(out)
        except KeyError as e:
            self.fail(f"SOT crashed with KeyError: {e}")

    def test_both_compiled_valid_output(self):
        """Both methods compiled — output must be a finite tensor."""
        m = ReproLayer()
        m.forward = paddle.jit.to_static(m.forward)
        m.step = paddle.jit.to_static(m.step)
        x = self._generate_input()
        out = m.step(x)
        self.assertFalse(paddle.isnan(out).any(), "Output contains NaN")
        self.assertFalse(paddle.isinf(out).any(), "Output contains Inf")

    def test_sub_module_alone(self):
        """Regression: sub-module access (self.conv, self.dim) must
        still work when only the callee is compiled."""
        m = ReproLayer()
        m.forward = paddle.jit.to_static(m.forward)
        x = self._generate_input()
        out = m.forward(x)
        self.assertEqual(out.shape, [2, 8, 32])


if __name__ == "__main__":
    unittest.main()
