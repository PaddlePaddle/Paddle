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

import os
import sys
import tempfile
import unittest
from pathlib import Path
from site import getsitepackages

import numpy as np
from setuptools import Distribution

import paddle
from paddle.utils.cpp_extension.cpp_extension import (
    _as_command_list,
    _is_ninja_available,
    _join_ninja_shell_list,
    _ninja_escape_path,
    _nt_quote_args,
    _write_ninja_file,
)
from paddle.utils.cpp_extension.extension_utils import (
    _get_all_paddle_includes_from_include_root,
)

# JIT compilation tests only run on Linux
# Helper function tests can run on all platforms
IS_LINUX = not (os.name == 'nt' or sys.platform.startswith('darwin'))

if IS_LINUX:
    from paddle.utils.cpp_extension import load


class TestNinjaHelperFunctions(unittest.TestCase):
    """Test helper functions for ninja compilation."""

    def test_is_ninja_available(self):
        result = _is_ninja_available()
        self.assertIsInstance(result, bool)
        # On Linux CI, ninja should be available
        if sys.platform.startswith('linux'):
            self.assertTrue(result, "ninja should be available on Linux CI")

    def test_ninja_escape_path(self):
        self.assertEqual(_ninja_escape_path("/path/to/file"), "/path/to/file")
        self.assertEqual(
            _ninja_escape_path("/path with space/file"),
            "/path$ with$ space/file",
        )
        self.assertEqual(
            _ninja_escape_path("C:/path/to/file"), "C$:/path/to/file"
        )
        self.assertEqual(
            _ninja_escape_path("/path/$var/file"), "/path/$$var/file"
        )
        self.assertEqual(
            _ninja_escape_path("C:/path with $var/file"),
            "C$:/path$ with$ $$var/file",
        )

    def test_nt_quote_args(self):
        self.assertEqual(_nt_quote_args(None), [])
        self.assertEqual(_nt_quote_args([]), [])
        self.assertEqual(_nt_quote_args(['-c', '-O2']), ['-c', '-O2'])
        result = _nt_quote_args(['/path with space/file', '-c'])
        self.assertEqual(result, ['"/path with space/file"', '-c'])
        result = _nt_quote_args(['"already quoted"', '-c'])
        self.assertEqual(result, ['"already quoted"', '-c'])

    def test_join_ninja_shell_list(self):
        self.assertEqual(
            _join_ninja_shell_list("simple string"), "simple string"
        )
        self.assertEqual(_join_ninja_shell_list([]), "")
        result = _join_ninja_shell_list(['-c', '-O2', '-I/usr/include'])
        # On Linux, uses shlex.join
        self.assertIn('-c', result)
        self.assertIn('-O2', result)
        result = _join_ninja_shell_list(['/path with space', '-c'])
        self.assertIn('/path with space', result)

    def test_as_command_list(self):
        self.assertEqual(_as_command_list("gcc"), ["gcc"])
        self.assertEqual(_as_command_list(["gcc", "-c"]), ["gcc", "-c"])
        self.assertEqual(_as_command_list(("gcc", "-c")), ["gcc", "-c"])

    def test_write_ninja_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ninja_path = os.path.join(tmpdir, "build.ninja")

            # Test writing content without newline
            content = "ninja_required_version = 1.5"
            _write_ninja_file(ninja_path, content)

            with open(ninja_path, 'r') as f:
                written_content = f.read()
            # Should add newline at end
            self.assertTrue(written_content.endswith('\n'))
            self.assertIn("ninja_required_version = 1.5", written_content)

            # Test writing content with newline
            content_with_newline = "rule compile\n  command = $cxx -c $in\n"
            _write_ninja_file(ninja_path, content_with_newline)

            with open(ninja_path, 'r') as f:
                written_content = f.read()
            self.assertTrue(written_content.endswith('\n'))

            # Test writing to nested directory
            nested_path = os.path.join(tmpdir, "nested", "dir", "build.ninja")
            _write_ninja_file(nested_path, "test content")
            self.assertTrue(os.path.exists(nested_path))


@unittest.skipIf(not IS_LINUX, "JIT compilation only supported on Linux")
class TestNinjaCompilation(unittest.TestCase):
    """Test ninja compilation for cpp extensions."""

    def setUp(self):
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.dtypes = ['float32', 'float64']

        # Get paddle include paths for Coverage CI
        self.paddle_includes = []
        for site_packages_path in getsitepackages():
            paddle_include_dir = Path(site_packages_path) / "paddle/include"
            self.paddle_includes.extend(
                _get_all_paddle_includes_from_include_root(
                    str(paddle_include_dir)
                )
            )
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.paddle_includes.append(test_dir)
        self.test_dir = test_dir

    def test_load_with_ninja_true(self):
        if not _is_ninja_available():
            self.skipTest("ninja is not available")

        sources = [os.path.join(self.test_dir, "custom_extension.cc")]

        ext = load(
            name='ninja_extension_true',
            sources=sources,
            extra_include_paths=self.paddle_includes,
            extra_cxx_cflags=['-w', '-g'],
            use_ninja=True,
            verbose=True,
        )
        np_x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        np_y = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        x = paddle.to_tensor(np_x, dtype='float32')
        y = paddle.to_tensor(np_y, dtype='float32')

        out = ext.custom_add(x, y)
        target_out = np.exp(np_x) + np.exp(np_y)
        np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

    def test_load_with_ninja_false(self):
        sources = [os.path.join(self.test_dir, "custom_extension.cc")]

        ext = load(
            name='ninja_extension_false',
            sources=sources,
            extra_include_paths=self.paddle_includes,
            extra_cxx_cflags=['-w', '-g'],
            use_ninja=False,
            verbose=True,
        )
        np_x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        np_y = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        x = paddle.to_tensor(np_x, dtype='float32')
        y = paddle.to_tensor(np_y, dtype='float32')

        out = ext.custom_add(x, y)
        target_out = np.exp(np_x) + np.exp(np_y)
        np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

    def test_load_ninja_auto_fallback(self):
        sources = [os.path.join(self.test_dir, "custom_extension.cc")]

        ext = load(
            name='ninja_extension_auto',
            sources=sources,
            extra_include_paths=self.paddle_includes,
            extra_cxx_cflags=['-w', '-g'],
            verbose=True,
        )

        np_x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        np_y = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        x = paddle.to_tensor(np_x, dtype='float32')
        y = paddle.to_tensor(np_y, dtype='float32')

        out = ext.custom_add(x, y)
        target_out = np.exp(np_x) + np.exp(np_y)
        np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

    def test_cuda_source_with_ninja(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not compiled")

        if not _is_ninja_available():
            self.skipTest("ninja is not available")

        sources = [
            os.path.join(self.test_dir, "custom_extension.cc"),
            os.path.join(self.test_dir, "custom_relu_forward.cu"),
        ]

        ext = load(
            name='ninja_cuda_extension',
            sources=sources,
            extra_include_paths=self.paddle_includes,
            extra_cxx_cflags=['-w', '-g'],
            use_ninja=True,
            verbose=True,
        )

        paddle.set_device('gpu')
        x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        x_tensor = paddle.to_tensor(x, dtype='float32')
        out = ext.relu_cuda_forward(x_tensor)
        pd_out = paddle.nn.functional.relu(x_tensor)
        np.testing.assert_allclose(out.numpy(), pd_out.numpy(), atol=1e-5)


class TestNinjaBuildExtension(unittest.TestCase):
    def _build_extension(self, **kwargs):
        from paddle.utils.cpp_extension.cpp_extension import BuildExtension

        return BuildExtension(dist=Distribution(), **kwargs)

    def test_use_ninja_attribute_default(self):
        build_ext = self._build_extension()
        self.assertTrue(build_ext.use_ninja, "use_ninja should default to True")

    def test_use_ninja_attribute_explicit_false(self):
        build_ext = self._build_extension(use_ninja=False)
        self.assertFalse(build_ext.use_ninja)

    def test_use_ninja_attribute_explicit_true(self):
        build_ext = self._build_extension(use_ninja=True)
        self.assertTrue(build_ext.use_ninja)

    def test_use_ninja_fallback_when_unavailable(self):
        # Create BuildExtension with use_ninja=True
        # If ninja is unavailable, it should fallback to False
        build_ext = self._build_extension(use_ninja=True)

        if not _is_ninja_available():
            self.assertFalse(
                build_ext.use_ninja,
                "use_ninja should be False when ninja is unavailable",
            )
        else:
            self.assertTrue(build_ext.use_ninja)


if __name__ == '__main__':
    unittest.main()
