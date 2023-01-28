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
import site
import sys
import unittest

import numpy as np

import paddle
from paddle.utils.cpp_extension.extension_utils import run_cmd


class TestCppExtensionSetupInstall(unittest.TestCase):
    """
    Tests setup install cpp extensions.
    """

    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # compile, install the custom op egg into site-packages under background
        if os.name == 'nt':
            cmd = 'cd /d {} && python cpp_extension_setup.py install'.format(
                cur_dir
            )
        else:
            cmd = 'cd {} && {} cpp_extension_setup.py install'.format(
                cur_dir, sys.executable
            )
        run_cmd(cmd)
        # os.system(cmd)

        # See: https://stackoverflow.com/questions/56974185/import-runtime-installed-module-using-pip-in-python-3
        if os.name == 'nt':
            site_dir = site.getsitepackages()[1]
        else:
            site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'custom_cpp_extension' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path
        )
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

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
        import custom_cpp_extension

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
        import custom_cpp_extension

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
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
