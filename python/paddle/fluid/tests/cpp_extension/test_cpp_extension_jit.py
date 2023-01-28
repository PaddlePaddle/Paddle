# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from site import getsitepackages

import numpy as np

import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

if os.name == 'nt' or sys.platform.startswith('darwin'):
    # only support Linux now
    exit()

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_cpp_extension\\custom_cpp_extension.pyd'.format(
    get_build_directory()
)
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

# Compile and load cpp extension Just-In-Time.
sources = ["custom_add.cc", "custom_sub.cc"]
paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include')
    )
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include', 'third_party')
    )
# include "custom_power.h"
paddle_includes.append(os.path.dirname(os.path.abspath(__file__)))

custom_cpp_extension = load(
    name='custom_cpp_extension',
    sources=sources,
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=['-w', '-g'],
    verbose=True,
)


class TestCppExtensionJITInstall(unittest.TestCase):
    """
    Tests setup install cpp extensions.
    """

    def setUp(self):
        # config seed
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        self.dtypes = ['float32', 'float64']

    def tearDown(self):
        pass

    def test_cpp_extension(self):
        self._test_extension_function()
        self._test_extension_class()

    def _test_extension_function(self):
        for dtype in self.dtypes:
            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            np_y = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            y = paddle.to_tensor(np_y, dtype=dtype)

            out = custom_cpp_extension.custom_add(x, y)
            target_out = np.exp(np_x) + np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

            # Test we can call a method not defined in the main C++ file.
            out = custom_cpp_extension.custom_sub(x, y)
            target_out = np.exp(np_x) - np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

    def _test_extension_class(self):
        for dtype in self.dtypes:
            # Test we can use CppExtension class with C++ methods.
            power = custom_cpp_extension.Power(3, 3)
            self.assertEqual(power.get().sum(), 9)
            self.assertEqual(power.forward().sum(), 9)

            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)

            power = custom_cpp_extension.Power(x)
            np.testing.assert_allclose(
                power.get().sum().numpy(), np.sum(np_x), atol=1e-5
            )
            np.testing.assert_allclose(
                power.forward().sum().numpy(),
                np.sum(np.power(np_x, 2)),
                atol=1e-5,
            )


if __name__ == '__main__':
    unittest.main()
