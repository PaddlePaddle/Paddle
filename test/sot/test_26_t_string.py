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

from paddle.jit.sot import psdb, symbolic_translate  # noqa: F401
from paddle.jit.sot.psdb import check_no_breakgraph  # noqa: F401
from paddle.jit.sot.utils.envs import strict_mode_guard
from paddle.jit.sot.utils.exceptions import InnerError

try:
    from string.templatelib import Interpolation, Template
except Exception:  # pragma: no cover - env without t-string support
    Interpolation = None
    Template = None


def _tstring_supported():
    if Template is None or Interpolation is None:
        return False
    try:
        compile("x = t'hello'", "<tstring>", "exec")
    except SyntaxError:
        return False
    return True


def _tstring_skip(*_args, **_kwargs):
    pass


if _tstring_supported():
    _tstring_ns = {}
    exec(
        """
@check_no_breakgraph
def test_t_literal():
    name = "world"
    return t"hello, {name}!"


@check_no_breakgraph
def test_t_with_expression():
    left = 2
    right = 5
    return t"{left} + {right} = {left + right}"


@check_no_breakgraph
def test_t_with_conversion_and_format():
    value = 3.14159
    return t"value={value!r:.2f}"


@check_no_breakgraph
def test_t_in_containers():
    prefix = "id"
    first = t"{prefix}-{1}"
    second = t"{prefix}-{2!a}"
    return [first, {"second": second}]


@check_no_breakgraph
def test_t_with_no_interpolation():
    return t"plain literal"


@check_no_breakgraph
def test_t_multiple_interpolations():
    a = 1
    b = 2
    c = 3
    return t"{a},{b},{c}"


@check_no_breakgraph
def test_t_with_format_spec_expression():
    value = 12.3456
    precision = 3
    return t"value={value:.{precision}f}"


@check_no_breakgraph
def test_t_with_ascii_conversion():
    text = "中文"
    return t"ascii={text!a}"


@check_no_breakgraph
def test_t_with_str_conversion():
    value = "hello"
    return t"str={value!s}"


def test_t_with_fallback_not_recursive():
    psdb.fallback(recursive=False)
    value = "hello"
    return t"fallback={value!s}"


def test_t_with_fallback_recursive():
    psdb.fallback(recursive=True)
    value = "hello"
    return t"fallback={value!s}"


@psdb.check_no_fallback
def test_t_with_forbidden_fallback():
    psdb.fallback(recursive=False)
    value = "hello"
    return t"fallback={value!s}"
""",
        globals(),
        _tstring_ns,
    )
    test_t_literal = _tstring_ns["test_t_literal"]
    test_t_with_expression = _tstring_ns["test_t_with_expression"]
    test_t_with_conversion_and_format = _tstring_ns[
        "test_t_with_conversion_and_format"
    ]
    test_t_in_containers = _tstring_ns["test_t_in_containers"]
    test_t_with_no_interpolation = _tstring_ns["test_t_with_no_interpolation"]
    test_t_multiple_interpolations = _tstring_ns[
        "test_t_multiple_interpolations"
    ]
    test_t_with_format_spec_expression = _tstring_ns[
        "test_t_with_format_spec_expression"
    ]
    test_t_with_ascii_conversion = _tstring_ns["test_t_with_ascii_conversion"]
    test_t_with_str_conversion = _tstring_ns["test_t_with_str_conversion"]
    test_t_with_fallback_not_recursive = _tstring_ns[
        "test_t_with_fallback_not_recursive"
    ]
    test_t_with_fallback_recursive = _tstring_ns[
        "test_t_with_fallback_recursive"
    ]
    test_t_with_forbidden_fallback = _tstring_ns[
        "test_t_with_forbidden_fallback"
    ]
else:
    test_t_literal = _tstring_skip
    test_t_with_expression = _tstring_skip
    test_t_with_conversion_and_format = _tstring_skip
    test_t_in_containers = _tstring_skip
    test_t_with_no_interpolation = _tstring_skip
    test_t_multiple_interpolations = _tstring_skip
    test_t_with_format_spec_expression = _tstring_skip
    test_t_with_ascii_conversion = _tstring_skip
    test_t_with_str_conversion = _tstring_skip
    test_t_with_fallback_not_recursive = _tstring_skip
    test_t_with_fallback_recursive = _tstring_skip
    test_t_with_forbidden_fallback = _tstring_skip


class TestTString(TestCaseBase):
    def _assert_tstring_like(self, actual, expected):
        if Template is None or Interpolation is None:
            self.skipTest(
                "Template strings are not supported by this interpreter."
            )

        self.assertIs(type(actual), type(expected))

        if isinstance(actual, Template):
            self._assert_tstring_like(actual.strings, expected.strings)
            self._assert_tstring_like(
                actual.interpolations, expected.interpolations
            )
            return

        if isinstance(actual, Interpolation):
            self._assert_tstring_like(actual.value, expected.value)
            self._assert_tstring_like(actual.expression, expected.expression)
            self.assertEqual(actual.conversion, expected.conversion)
            self.assertEqual(actual.format_spec, expected.format_spec)
            return

        if isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a_item, e_item in zip(actual, expected):
                self._assert_tstring_like(a_item, e_item)
            return

        if isinstance(actual, dict):
            self.assertEqual(set(actual.keys()), set(expected.keys()))
            for key in actual:
                self._assert_tstring_like(actual[key], expected[key])
            return

        if isinstance(actual, set):
            self.assertEqual(actual, expected)
            return

        # Fallback to the generic nested matcher for tensors / arrays / scalars.
        self.assert_nest_match(actual, expected)

    def assert_tstring_results(self, func, *args, **kwargs):
        sym_output = symbolic_translate(func)(*args, **kwargs)
        paddle_output = func(*args, **kwargs)
        self._assert_tstring_like(sym_output, paddle_output)

    def test_symbolic_translate_handles_t_string(self):
        self.assert_tstring_results(test_t_literal)

    def test_symbolic_translate_handles_formats_and_containers(self):
        self.assert_tstring_results(test_t_with_expression)
        self.assert_tstring_results(test_t_with_conversion_and_format)
        self.assert_tstring_results(test_t_in_containers)

    def test_symbolic_translate_handles_more_cases(self):
        self.assert_tstring_results(test_t_with_no_interpolation)
        self.assert_tstring_results(test_t_multiple_interpolations)
        self.assert_tstring_results(test_t_with_format_spec_expression)
        self.assert_tstring_results(test_t_with_ascii_conversion)
        self.assert_tstring_results(test_t_with_str_conversion)

    @strict_mode_guard(False)
    def test_tstring_fallback_not_recursive(self):
        self.assert_tstring_results(test_t_with_fallback_not_recursive)

    @strict_mode_guard(False)
    def test_tstring_fallback_recursive(self):
        self.assert_tstring_results(test_t_with_fallback_recursive)

    def test_tstring_check_no_fallback(self):
        if not _tstring_supported():
            self.skipTest(
                "Template strings are not supported by this interpreter."
            )
        with self.assertRaises(InnerError):
            symbolic_translate(test_t_with_forbidden_fallback)()


if __name__ == "__main__":
    unittest.main()
