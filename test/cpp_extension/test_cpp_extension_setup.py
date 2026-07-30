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
from utils import check_output

import paddle
from paddle.utils.cpp_extension.extension_utils import run_cmd


class TestCppExtensionSetupInstall(unittest.TestCase):
    """
    Tests setup install cpp extensions.
    """

    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # install general extension
        # compile, install the custom op egg into site-packages under background
        site_dir = site.getsitepackages()[0]
        cmd = f'cd {cur_dir} && {sys.executable} cpp_extension_setup.py install'
        if os.name != 'nt':
            cmd += f' --install-lib={site_dir}'
        run_cmd(cmd)

        custom_install_path = [
            x for x in os.listdir(site_dir) if 'custom_cpp_extension' in x
        ]

        assert len(custom_install_path) == 2, (
            f"Matched egg number is {len(custom_install_path)}."
        )

        sys.path.append(os.path.join(site_dir, custom_install_path[0]))
        #################################

        # config seed
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        self.dtypes = ['float32', 'float64']

    def tearDown(self):
        pass

    def test_cpp_extension(self):
        self._test_extension_function_plain()
        self._test_vector_tensor()
        self._test_extension_class()
        self._test_nullable_tensor()
        self._test_optional_tensor()
        self._test_scalar_type_caster()

    def _test_extension_function_plain(self):
        import custom_cpp_extension

        for dtype in self.dtypes:
            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            np_y = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            y = paddle.to_tensor(np_y, dtype=dtype)
            # Test custom_cpp_extension
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
            # Test custom_cpp_extension
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

    def _test_vector_tensor(self):
        import custom_cpp_extension

        for dtype in self.dtypes:
            np_inputs = [
                np.random.uniform(-1, 1, [4, 8]).astype(dtype) for _ in range(3)
            ]
            inputs = [paddle.to_tensor(np_x, dtype=dtype) for np_x in np_inputs]

            out = custom_cpp_extension.custom_tensor(inputs)
            target_out = [x + 1 for x in inputs]
            for i in range(3):
                np.testing.assert_allclose(
                    out[i].numpy(), target_out[i].numpy(), atol=1e-5
                )

    def _test_nullable_tensor(self):
        import custom_cpp_extension

        x = custom_cpp_extension.nullable_tensor(True)
        assert x is None, "Return None when input parameter return_none = True"
        x = custom_cpp_extension.nullable_tensor(False).numpy()
        x_np = np.ones(shape=[2, 2])
        np.testing.assert_array_equal(
            x,
            x_np,
            err_msg=f'extension out: {x},\n numpy out: {x_np}',
        )

    def _test_optional_tensor(self):
        import custom_cpp_extension

        x = custom_cpp_extension.optional_tensor(True)
        assert x is None, (
            "Return None when input parameter return_option = True"
        )
        x = custom_cpp_extension.optional_tensor(False).numpy()
        x_np = np.ones(shape=[2, 2])
        np.testing.assert_array_equal(
            x,
            x_np,
            err_msg=f'extension out: {x},\n numpy out: {x_np}',
        )

    def _test_scalar_type_caster(self):
        import custom_cpp_extension

        expected_values = {
            paddle.uint8: 0,
            paddle.int8: 1,
            paddle.int16: 2,
            paddle.int32: 3,
            paddle.int64: 4,
            paddle.float16: 5,
            paddle.float32: 6,
            paddle.float64: 7,
            paddle.complex64: 9,
            paddle.complex128: 10,
            paddle.bool: 11,
            paddle.bfloat16: 15,
            paddle.float8_e5m2: 23,
            paddle.float8_e4m3fn: 24,
            paddle.uint16: 27,
            paddle.uint32: 28,
        }
        for dtype, expected_value in expected_values.items():
            self.assertIs(
                custom_cpp_extension.scalar_type_round_trip(dtype), dtype
            )
            self.assertEqual(
                custom_cpp_extension.scalar_type_value(dtype),
                expected_value,
            )

    def _test_cuda_relu(self):
        import custom_cpp_extension

        paddle.set_device('gpu')
        x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        x = paddle.to_tensor(x, dtype='float32')
        out = custom_cpp_extension.relu_cuda_forward(x)
        pd_out = paddle.nn.functional.relu(x)
        check_output(out, pd_out, "out")


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        sys.exit()
    unittest.main()
