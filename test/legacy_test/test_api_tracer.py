# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import tempfile
import unittest

import numpy as np
import yaml

import paddle
from paddle.api_tracer import start_api_tracer, stop_api_tracer
from paddle.api_tracer.api_tracer import (
    ConfigDump,
    _hooked_apis,
    _originals,
    expand_wildcard,
)


class TestDumpPlace(unittest.TestCase):
    """Test the new Place branch in dump_item_str."""

    def test_cpu_place(self):
        dumper = ConfigDump()
        result = dumper.dump_item_str("test", paddle.CPUPlace())
        self.assertEqual(result, "Place(cpu)")


class TestExpandWildcard(unittest.TestCase):
    """Test expand_wildcard for plain APIs and wildcard patterns."""

    def test_plain_api(self):
        result = expand_wildcard("paddle.abs")
        self.assertEqual(result, ["paddle.abs"])

    def test_wildcard(self):
        result = expand_wildcard("math.*")
        self.assertIn("math.sin", result)
        self.assertIn("math.cos", result)
        for api in result:
            name = api.split(".")[-1]
            self.assertFalse(name.startswith("_"))


class TestStartStopApiTracer(unittest.TestCase):
    """Test start_api_tracer / stop_api_tracer hook lifecycle."""

    def setUp(self):
        self.save_dir = tempfile.mkdtemp()
        self.api_yaml = os.path.join(self.save_dir, "apis.yaml")
        self.trace_log = os.path.join(self.save_dir, "trace.log")

    def tearDown(self):
        if _hooked_apis:
            stop_api_tracer()
        shutil.rmtree(self.save_dir)

    def _write_yaml(self, apis):
        with open(self.api_yaml, "w") as f:
            yaml.dump({"apis": apis}, f)

    def test_hook_and_unhook(self):
        self._write_yaml(["paddle.abs"])
        original_abs = paddle.abs

        start_api_tracer(self.api_yaml, self.trace_log)
        self.assertIn("paddle.abs", _hooked_apis)
        self.assertNotEqual(paddle.abs, original_abs)

        stop_api_tracer()
        self.assertIs(paddle.abs, original_abs)
        self.assertEqual(len(_hooked_apis), 0)
        self.assertEqual(len(_originals), 0)

    def test_trace_output_written(self):
        self._write_yaml(["paddle.abs"])

        start_api_tracer(self.api_yaml, self.trace_log)
        paddle.abs(paddle.to_tensor([-1.0, 2.0]))
        stop_api_tracer()

        with open(self.trace_log, "r") as f:
            content = f.read()
        self.assertIn("paddle.abs(", content)

    def test_wildcard_hook(self):
        self._write_yaml(["paddle._C_ops.*"])

        start_api_tracer(self.api_yaml, self.trace_log)
        self.assertGreater(len(_hooked_apis), 0)

        stop_api_tracer()
        self.assertEqual(len(_hooked_apis), 0)

    def test_empty_yaml(self):
        self._write_yaml([])

        start_api_tracer(self.api_yaml, self.trace_log)
        self.assertEqual(len(_hooked_apis), 0)
        stop_api_tracer()

    def test_recursion_guard(self):
        """paddle.abs internally calls paddle._C_ops.abs.
        With recursion guard, only paddle.abs should appear in trace."""
        self._write_yaml(["paddle.abs", "paddle._C_ops.abs"])

        start_api_tracer(self.api_yaml, self.trace_log)
        result = paddle.abs(paddle.to_tensor([-3.0]))
        np.testing.assert_allclose(result.numpy(), [3.0])
        stop_api_tracer()

        with open(self.trace_log, "r") as f:
            lines = f.readlines()
        apis_traced = [l.split("(")[0] for l in lines]
        self.assertIn("paddle.abs", apis_traced)
        self.assertNotIn("paddle._C_ops.abs", apis_traced)


class TestErrorHandling(unittest.TestCase):
    """Test exception handling code paths in api_tracer."""

    def setUp(self):
        self.save_dir = tempfile.mkdtemp()
        self.api_yaml = os.path.join(self.save_dir, "apis.yaml")
        self.trace_log = os.path.join(self.save_dir, "trace.log")

    def tearDown(self):
        if _hooked_apis:
            stop_api_tracer()
        shutil.rmtree(self.save_dir)

    def test_nonexistent_api_skipped(self):
        """start_api_tracer should skip APIs that cannot be resolved
        (eval/getattr failure) instead of raising."""
        with open(self.api_yaml, "w") as f:
            yaml.dump(
                {
                    "apis": [
                        "paddle.nonexistent_api_xyz",
                        "no_such_module.func",
                    ]
                },
                f,
            )
        start_api_tracer(self.api_yaml, self.trace_log)
        self.assertNotIn("paddle.nonexistent_api_xyz", _hooked_apis)
        self.assertNotIn("no_such_module.func", _hooked_apis)
        stop_api_tracer()

    def test_expand_wildcard_invalid_module(self):
        """expand_wildcard should return [] for a non-existent module."""
        result = expand_wildcard("nonexistent_module_xyz.*")
        self.assertEqual(result, [])

    def test_dump_item_str_unrecognized_type(self):
        """dump_item_str should return '' for an unrecognized type."""
        dumper = ConfigDump()
        result = dumper.dump_item_str("test_api", object())
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()
