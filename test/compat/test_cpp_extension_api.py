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
import unittest

import paddle.base as core
from paddle.utils.cpp_extension import (
    CUDA_HOME,
    _get_cuda_arch_flags,
    _get_num_workers,
    _get_pybind11_abi_build_flags,
)


@unittest.skipIf(not core.is_compiled_with_cuda(), 'should compile with cuda.')
class TestGetCudaArchFlags(unittest.TestCase):
    def setUp(self):
        self._old_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_with_user_cflags(self):
        flags = _get_cuda_arch_flags(cflags=["-arch=sm_90"])
        self.assertEqual(flags, [])

    def test_with_env_hopper(self):
        os.environ["PADDLE_CUDA_ARCH_LIST"] = "Hopper"
        flags = _get_cuda_arch_flags()
        # Hopper -> 9.0+PTX -> sm_90 + compute_90
        self.assertIn("-gencode=arch=compute_90,code=sm_90", flags)
        self.assertIn("-gencode=arch=compute_90,code=compute_90", flags)

    def test_with_env_hopper_and_flags(self):
        os.environ["PADDLE_CUDA_ARCH_LIST"] = "Hopper"
        flags = _get_cuda_arch_flags("Hopper")
        # Hopper -> 9.0+PTX -> sm_90 + compute_90
        self.assertIn("-gencode=arch=compute_90,code=sm_90", flags)
        self.assertIn("-gencode=arch=compute_90,code=compute_90", flags)

    def test_with_env_multiple(self):
        os.environ["PADDLE_CUDA_ARCH_LIST"] = "8.6;9.0+PTX"
        flags = _get_cuda_arch_flags()
        self.assertIn("-gencode=arch=compute_86,code=sm_86", flags)
        self.assertIn("-gencode=arch=compute_90,code=sm_90", flags)
        self.assertIn("-gencode=arch=compute_90,code=compute_90", flags)

    def test_auto_detect(self):
        if "PADDLE_CUDA_ARCH_LIST" in os.environ:
            del os.environ["PADDLE_CUDA_ARCH_LIST"]
        flags = _get_cuda_arch_flags()
        self.assertTrue(len(flags) > 0)

    def test_get_cuda_arch_flags_with_invalid_arch(self):
        os.environ["PADDLE_CUDA_ARCH_LIST"] = "invalid_arch"
        with self.assertRaises(ValueError) as context:
            _get_cuda_arch_flags()
        self.assertIn(
            "Unknown CUDA arch (invalid_arch) or GPU not supported",
            str(context.exception),
        )

    def test_skip_paddle_extension_name_flag(self):
        flags = _get_cuda_arch_flags(cflags=["-DPADDLE_EXTENSION_NAME=my_ext"])
        self.assertNotEqual(flags, [])


class TestCppExtensionUtils(unittest.TestCase):
    def test_cuda_home(self):
        if core.is_compiled_with_cuda():
            value = CUDA_HOME
            self.assertTrue(value is None or isinstance(value, str))

    def test_get_pybind11_abi_build_flags(self):
        flags = _get_pybind11_abi_build_flags()
        self.assertIsInstance(flags, list)
        for f in flags:
            self.assertIsInstance(f, str)

    def test_get_num_workers_with_env_verbose_false(self):
        os.environ["MAX_JOBS"] = "8"
        num = _get_num_workers(verbose=False)
        self.assertEqual(num, 8)

    def test_get_num_workers_with_env_verbose_true(self):
        os.environ["MAX_JOBS"] = "8"
        num = _get_num_workers(verbose=True)
        self.assertEqual(num, 8)

    def test_get_num_workers_without_env_verbose_true(self):
        if "MAX_JOBS" in os.environ:
            del os.environ["MAX_JOBS"]
        num = _get_num_workers(verbose=True)
        self.assertEqual(num, None)


if __name__ == "__main__":
    unittest.main()
