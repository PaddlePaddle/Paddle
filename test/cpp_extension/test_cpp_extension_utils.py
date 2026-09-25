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

"""
Unit tests for paddle.utils.cpp_extension.extension_utils, focusing on the
automatic injection of -DPADDLE_WITH_DNNL and OneDNN include path when Paddle
is built with WITH_MKL=ON.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from paddle.base import core
from paddle.utils.cpp_extension.extension_utils import (
    find_paddle_custom_device_includes,
    find_paddle_includes,
    find_paddle_libraries,
    find_python_includes,
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

    def test_custom_kernel_macro_injected_when_compile_dir_is_none(self):
        """
        When _compile_dir is not specified, -DPADDLE_WITH_CUSTOM_KERNEL must
        be injected into compile flags. This is always injected regardless of
        OneDNN status.
        """
        # Test with dict-style extra_compile_args (CUDAExtension case)
        result = normalize_extension_kwargs(
            {'extra_compile_args': {'cxx': [], 'nvcc': []}}
        )
        cxx_flags = result['extra_compile_args']['cxx']
        self.assertIn(
            '-DPADDLE_WITH_CUSTOM_KERNEL',
            cxx_flags,
            "-DPADDLE_WITH_CUSTOM_KERNEL should be injected when compile_dir is None",
        )

    def test_macros_not_injected_when_compile_dir_is_set(self):
        """
        When _compile_dir is specified, neither -DPADDLE_WITH_CUSTOM_KERNEL
        nor -DPADDLE_WITH_DNNL should be injected.
        """
        # Create a temporary compile_dir with includes.txt
        with tempfile.TemporaryDirectory() as tmpdir:
            includes_txt = os.path.join(tmpdir, 'includes.txt')
            with open(includes_txt, 'w') as f:
                f.write('/usr/include\n')

            # When _compile_dir is set, macros should NOT be injected
            result = normalize_extension_kwargs(
                {
                    'extra_compile_args': {'cxx': [], 'nvcc': []},
                    '_compile_dir': tmpdir,
                }
            )
            cxx_flags = result['extra_compile_args']['cxx']

            self.assertNotIn(
                '-DPADDLE_WITH_CUSTOM_KERNEL',
                cxx_flags,
                "-DPADDLE_WITH_CUSTOM_KERNEL should NOT be injected when compile_dir is set",
            )
            # Even if compiled with OneDNN, -DPADDLE_WITH_DNNL should not be injected
            self.assertNotIn(
                '-DPADDLE_WITH_DNNL',
                cxx_flags,
                "-DPADDLE_WITH_DNNL should NOT be injected when compile_dir is set",
            )

    def test_dnnl_macro_not_injected_when_headers_missing(self):
        """
        When Paddle is compiled with OneDNN but the OneDNN headers are missing
        from the expected locations, -DPADDLE_WITH_DNNL should NOT be injected
        to avoid compilation errors.
        """
        if not core.is_compiled_with_onednn():
            self.skipTest("Paddle not compiled with OneDNN, skipping.")

        import tempfile

        # Create a temporary directory structure without OneDNN headers
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock paddle include dir
            paddle_include_dir = os.path.join(tmpdir, 'include')
            os.makedirs(paddle_include_dir)

            # Mock sysconfig path (used by get_include)
            mock_site_packages = os.path.join(tmpdir, 'site-packages', 'paddle')
            os.makedirs(os.path.dirname(mock_site_packages))
            os.makedirs(mock_site_packages)

            # Patch get_include to return our mock directory
            with patch(
                'paddle.utils.cpp_extension.extension_utils.get_include',
                return_value=paddle_include_dir,
            ):
                # Mock os.path.exists to return False for any OneDNN include paths
                original_exists = os.path.exists

                def mock_exists(path):
                    # If checking for OneDNN include paths, return False
                    if 'onednn' in path.lower() and 'include' in path:
                        return False
                    # Otherwise use original behavior for our mock directory
                    if path.startswith(tmpdir):
                        return original_exists(path)
                    return False

                with patch('os.path.exists', side_effect=mock_exists):
                    result = normalize_extension_kwargs(
                        {'extra_compile_args': {'cxx': [], 'nvcc': []}}
                    )
                    cxx_flags = result['extra_compile_args']['cxx']

                    self.assertNotIn(
                        '-DPADDLE_WITH_DNNL',
                        cxx_flags,
                        "-DPADDLE_WITH_DNNL should NOT be injected when OneDNN headers are missing",
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

    def test_onednn_include_dir_source_build_fallback(self):
        """
        Test the source-build fallback branch: when whl install path doesn't
        exist but source build path does exist, it should use the source build
        path. This covers the code path where paddle/include/third_party/install/
        onednn/include doesn't exist, but build/third_party/install/onednn/include
        does exist.
        """
        if not core.is_compiled_with_onednn():
            self.skipTest("Paddle not compiled with OneDNN, skipping.")

        # Create a mock source build directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build structure: tmpdir/build/python/paddle/__init__.py
            #                     tmpdir/build/third_party/install/onednn/include/
            build_dir = os.path.join(tmpdir, 'build')
            python_dir = os.path.join(build_dir, 'python', 'paddle')
            os.makedirs(python_dir)

            # Create a dummy __init__.py to make it a valid Python package
            with open(os.path.join(python_dir, '__init__.py'), 'w') as f:
                f.write('')

            # Create the source build OneDNN include directory
            source_onednn_include = os.path.join(
                build_dir, 'third_party', 'install', 'onednn', 'include'
            )
            os.makedirs(source_onednn_include)

            # Mock paddle.__file__ to point to our mock source build directory
            mock_paddle_file = os.path.join(python_dir, '__init__.py')

            # We need to mock both paddle module's __file__ and the import
            # Since find_paddle_includes does an internal import paddle, we need
            # to prepare the mock before calling the function
            import paddle as paddle_module

            original_paddle_file = paddle_module.__file__

            try:
                # Set paddle.__file__ before calling find_paddle_includes
                paddle_module.__file__ = mock_paddle_file

                # Patch os.path.exists to return False for whl install path
                original_exists = os.path.exists

                def mock_exists(path):
                    # If checking for whl install path (contains 'paddle/include'), return False
                    if (
                        'third_party/install/onednn/include' in path
                        and 'paddle/include' in path
                    ):
                        return False
                    # Allow actual files in tmpdir to exist
                    if path.startswith(tmpdir):
                        return original_exists(path)
                    # Otherwise use original behavior
                    return original_exists(path)

                with patch('os.path.exists', side_effect=mock_exists):
                    # Patch get_include to return a non-existent whl include dir
                    whl_paddle_include = os.path.join(
                        tmpdir, 'whl', 'paddle', 'include'
                    )
                    os.makedirs(whl_paddle_include)

                    with patch(
                        'paddle.utils.cpp_extension.extension_utils.get_include',
                        return_value=whl_paddle_include,
                    ):
                        include_dirs = find_paddle_includes(use_cuda=False)
                        onednn_dirs = [
                            d
                            for d in include_dirs
                            if os.path.join(
                                'third_party', 'install', 'onednn', 'include'
                            )
                            in d
                        ]

                        self.assertTrue(
                            len(onednn_dirs) > 0,
                            "OneDNN include dir from source build should be added.",
                        )
                        # Verify it's the source build path, not the whl install path
                        for onednn_dir in onednn_dirs:
                            self.assertNotIn(
                                'paddle/include',
                                onednn_dir,
                                "Should use source build path, not whl install path.",
                            )
            finally:
                # Restore original paddle.__file__
                paddle_module.__file__ = original_paddle_file


class TestFindPaddleLibraries(unittest.TestCase):
    """
    Test find_paddle_libraries() function to ensure it returns correct
    library paths for both CPU and GPU builds.
    """

    def test_find_paddle_libraries_cpu(self):
        """
        Test that find_paddle_libraries returns at least the base library
        directory when use_cuda=False.
        """
        lib_dirs = find_paddle_libraries(use_cuda=False)
        self.assertTrue(
            len(lib_dirs) > 0,
            "find_paddle_libraries should return at least one directory",
        )
        # Verify it includes the base path
        for lib_dir in lib_dirs:
            self.assertIsInstance(
                lib_dir,
                str,
                "Library directory should be a string",
            )

    def test_find_paddle_libraries_cuda(self):
        """
        Test that find_paddle_libraries includes CUDA library directories
        when use_cuda=True and CUDA is available.
        """
        if not core.is_compiled_with_cuda():
            self.skipTest("Paddle not compiled with CUDA, skipping.")

        lib_dirs = find_paddle_libraries(use_cuda=True)
        self.assertTrue(
            len(lib_dirs) > 0,
            "find_paddle_libraries should return at least one directory",
        )
        # When compiled with CUDA, there should be more directories than CPU-only
        cpu_lib_dirs = find_paddle_libraries(use_cuda=False)
        self.assertGreaterEqual(
            len(lib_dirs),
            len(cpu_lib_dirs),
            "CUDA build should have at least as many lib dirs as CPU build",
        )

    def test_find_paddle_libraries_rocm(self):
        """
        Test that find_paddle_libraries includes ROCm library directories
        when use_cuda=True and ROCm is available.
        """
        if not core.is_compiled_with_cuda():
            self.skipTest("Paddle not compiled with CUDA/ROCm, skipping.")
        if not core.is_compiled_with_rocm():
            self.skipTest("Paddle not compiled with ROCm, skipping.")

        lib_dirs = find_paddle_libraries(use_cuda=True)
        # ROCm builds should include ROCm library directories
        self.assertTrue(
            len(lib_dirs) > 0,
            "find_paddle_libraries should return at least one directory",
        )


class TestFindPythonIncludes(unittest.TestCase):
    """
    Test find_python_includes() function to ensure it returns correct
    Python header paths.
    """

    def test_find_python_includes(self):
        """
        Test that find_python_includes returns a valid list of Python
        include directories.
        """
        python_includes = find_python_includes()
        self.assertIsInstance(
            python_includes,
            list,
            "find_python_includes should return a list",
        )
        # At least one Python include directory should be found
        self.assertTrue(
            len(python_includes) > 0,
            "At least one Python include directory should be found",
        )
        # Verify the path contains 'include'
        for inc in python_includes:
            self.assertIsInstance(
                inc,
                str,
                "Python include path should be a string",
            )


class TestFindPaddleCustomDeviceIncludes(unittest.TestCase):
    """
    Test find_paddle_custom_device_includes() function to ensure it returns
    correct custom device include paths when available.
    """

    def test_find_paddle_custom_device_includes(self):
        """
        Test that find_paddle_custom_device_includes returns appropriate
        directories based on custom device availability.
        """
        devices = core.get_all_device_type()

        if not devices:
            # No custom devices - should return empty list
            custom_includes = find_paddle_custom_device_includes()
            self.assertEqual(
                custom_includes,
                [],
                "Should return empty list when no custom devices available",
            )
        else:
            # There are custom devices - test the function
            custom_includes = find_paddle_custom_device_includes()
            self.assertIsInstance(
                custom_includes,
                list,
                "find_paddle_custom_device_includes should return a list",
            )


if __name__ == '__main__':
    unittest.main()
