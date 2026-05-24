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

"""
Unit tests for paddle.utils.cpp_extension.extension_utils, focusing on the
automatic injection of -DPADDLE_WITH_DNNL and OneDNN include path when Paddle
is built with WITH_MKL=ON.
"""

import os
import unittest

from paddle.base import core
from paddle.utils.cpp_extension.extension_utils import (
    find_paddle_includes,
    normalize_extension_kwargs,
)


class TestOneDNNFlagsInjection(unittest.TestCase):
    """
    Verify that normalize_extension_kwargs() automatically appends
    -DPADDLE_WITH_DNNL to both cxx and nvcc compile flags when Paddle is
    compiled with OneDNN (WITH_MKL=ON), and does NOT inject it otherwise.
    """

    def _get_flags(self, kwargs):
        """Run normalize_extension_kwargs and return (cxx_flags, nvcc_flags)."""
        result = normalize_extension_kwargs(kwargs)
        args = result['extra_compile_args']
        if isinstance(args, dict):
            return args.get('cxx', []), args.get('nvcc', [])
        return args, args

    def test_dnnl_macro_injected_when_onednn_enabled(self):
        """
        When is_compiled_with_onednn() is True, -DPADDLE_WITH_DNNL must
        appear in both cxx and nvcc flags (injected via add_compile_flag).
        """
        if not core.is_compiled_with_onednn():
            self.skipTest("Paddle not compiled with OneDNN, skipping.")

        # dict-style extra_compile_args (the common CUDAExtension case)
        cxx_flags, nvcc_flags = self._get_flags(
            {'extra_compile_args': {'cxx': [], 'nvcc': []}}
        )
        self.assertIn(
            '-DPADDLE_WITH_DNNL',
            cxx_flags,
            "-DPADDLE_WITH_DNNL should be injected into cxx flags on DNNL build",
        )
        self.assertIn(
            '-DPADDLE_WITH_DNNL',
            nvcc_flags,
            "-DPADDLE_WITH_DNNL should be injected into nvcc flags on DNNL build",
        )

    def test_dnnl_macro_injected_list_style(self):
        """
        Same check for list-style extra_compile_args (the CppExtension case).
        """
        if not core.is_compiled_with_onednn():
            self.skipTest("Paddle not compiled with OneDNN, skipping.")

        cxx_flags, _ = self._get_flags({'extra_compile_args': []})
        self.assertIn(
            '-DPADDLE_WITH_DNNL',
            cxx_flags,
            "-DPADDLE_WITH_DNNL should be injected into list-style flags on DNNL build",
        )

    def test_dnnl_macro_not_injected_when_onednn_disabled(self):
        """
        When is_compiled_with_onednn() is False, -DPADDLE_WITH_DNNL must NOT
        be injected (do not pollute non-DNNL builds).
        """
        if core.is_compiled_with_onednn():
            self.skipTest("Paddle compiled with OneDNN, skipping.")

        cxx_flags, nvcc_flags = self._get_flags(
            {'extra_compile_args': {'cxx': [], 'nvcc': []}}
        )
        self.assertNotIn(
            '-DPADDLE_WITH_DNNL',
            cxx_flags,
            "-DPADDLE_WITH_DNNL should NOT be injected on non-DNNL build",
        )
        self.assertNotIn(
            '-DPADDLE_WITH_DNNL',
            nvcc_flags,
            "-DPADDLE_WITH_DNNL should NOT be injected on non-DNNL build",
        )

    def test_existing_flags_preserved(self):
        """
        Pre-existing user flags must not be removed when DNNL flag is injected.
        """
        if not core.is_compiled_with_onednn():
            self.skipTest("Paddle not compiled with OneDNN, skipping.")

        user_flags = ['-O3', '-DSOME_USER_MACRO']
        cxx_flags, _ = self._get_flags(
            {'extra_compile_args': {'cxx': list(user_flags), 'nvcc': []}}
        )
        for flag in user_flags:
            self.assertIn(
                flag,
                cxx_flags,
                f"User flag {flag} should be preserved after DNNL injection",
            )


class TestOneDNNIncludePath(unittest.TestCase):
    """
    Verify that find_paddle_includes() automatically appends the bundled
    OneDNN include directory when Paddle is compiled with WITH_MKL=ON.
    """

    def test_onednn_include_dir_added_when_enabled(self):
        """
        When is_compiled_with_onednn() is True, the include list must contain
        a path ending with third_party/install/onednn/include (if it exists
        inside the Paddle installation).
        """
        if not core.is_compiled_with_onednn():
            self.skipTest("Paddle not compiled with OneDNN, skipping.")

        include_dirs = find_paddle_includes(use_cuda=False)
        onednn_dirs = [
            d
            for d in include_dirs
            if os.path.join('third_party', 'install', 'onednn', 'include') in d
        ]

        # The path is only appended when it exists on disk, so we first check
        # whether the expected directory is present in the installation.
        import paddle as _paddle

        paddle_include = os.path.join(
            os.path.dirname(_paddle.__file__), 'include'
        )
        expected_dir = os.path.join(
            paddle_include,
            'third_party',
            'install',
            'onednn',
            'include',
        )

        if os.path.exists(expected_dir):
            self.assertTrue(
                len(onednn_dirs) > 0,
                f"OneDNN include dir '{expected_dir}' exists on disk but "
                "was not added to include_dirs by find_paddle_includes().",
            )
            self.assertIn(expected_dir, include_dirs)
        else:
            # Directory absent in this environment -- the code should silently
            # skip it (no error, no stale path appended).
            self.assertEqual(
                len(onednn_dirs),
                0,
                "Non-existent OneDNN include dir should not be appended.",
            )

    def test_onednn_include_dir_not_added_when_disabled(self):
        """
        When is_compiled_with_onednn() is False, no OneDNN path should appear
        in the include list.
        """
        if core.is_compiled_with_onednn():
            self.skipTest("Paddle compiled with OneDNN, skipping.")

        include_dirs = find_paddle_includes(use_cuda=False)
        onednn_dirs = [d for d in include_dirs if 'onednn' in d.lower()]
        self.assertEqual(
            len(onednn_dirs),
            0,
            "No OneDNN include dir should appear on non-DNNL builds.",
        )


if __name__ == '__main__':
    unittest.main()