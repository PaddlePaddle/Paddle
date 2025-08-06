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

import unittest

import numpy as np

import paddle


class TensorDtypeConversionsTest(unittest.TestCase):
    """
    Unit tests for all supported tensor dtype conversion methods.
    """

    _supported_dtype_conversions = {
        # float
        'float16': 'float16',
        'half': 'float16',
        'bfloat16': 'bfloat16',
        'float32': 'float32',
        'float': 'float32',
        'float64': 'float64',
        'double': 'float64',
        # int
        'int8': 'int8',
        'char': 'int8',
        'uint8': 'uint8',
        'byte': 'uint8',
        'int16': 'int16',
        'short': 'int16',
        'int32': 'int32',
        'int': 'int32',
        'int64': 'int64',
        'long': 'int64',
        # other
        'bool': 'bool',
        'complex64': 'complex64',
        'complex128': 'complex128',
        'cfloat': 'complex64',
        'cdouble': 'complex128',
    }
    _device = paddle.device.get_device()
    _total_init_dtype = [
        'float16',
        'float32',
        'float64',
        'int8',
        'uint8',
        'int16',
        'int32',
        'int64',
        'bool',
        'complex64',
        'complex128',
    ]

    def setUp(self):
        self.shape = [10, 1000]

    def _get_paddle_dtype(self, dtype_str):
        """Get the Paddle dtype constant by string name."""
        return getattr(paddle, dtype_str)

    def test_bfloat16_conversion(self):
        for init_dtype in self._total_init_dtype:
            if self._device.startswith('xpu') and init_dtype == 'complex128':
                continue
            tensor = paddle.randn(self.shape).astype(init_dtype)
            converted_tensor = tensor.bfloat16()
            self.assertEqual(converted_tensor.dtype, paddle.bfloat16)
            self.assertEqual(converted_tensor.shape, tensor.shape)

        for (
            method_name,
            target_dtype,
        ) in self._supported_dtype_conversions.items():
            if self._device.startswith('xpu') and target_dtype == 'complex128':
                continue
            tensor = paddle.randn(self.shape).astype('bfloat16')
            converted_tensor = getattr(tensor, method_name)()
            self.assertEqual(
                converted_tensor.dtype, self._get_paddle_dtype(target_dtype)
            )
            self.assertEqual(converted_tensor.shape, tensor.shape)

    def test_all_dtype_conversions(self):
        """Test all dtype conversion methods."""
        for (
            method_name,
            target_dtype,
        ) in self._supported_dtype_conversions.items():
            if target_dtype == 'bfloat16':
                continue
            for init_dtype in self._total_init_dtype:
                if self._device.startswith('xpu') and (
                    target_dtype == 'complex128' or init_dtype == 'complex128'
                ):
                    self.skipTest("Skipping complex conversion tests on XPU")

                with self.subTest(
                    method=method_name,
                    init_dtype=init_dtype,
                    target_dtype=target_dtype,
                ):
                    self._test_single_dtype_conversion(
                        method_name, init_dtype, target_dtype
                    )

    def _test_single_dtype_conversion(
        self, method_name, init_dtype, target_dtype
    ):
        """Test a single dtype conversion method."""
        if init_dtype.startswith('float'):
            data_np = np.random.randn(*self.shape).astype(init_dtype)
        elif init_dtype.startswith('complex'):
            data_np_real = np.random.randn(*self.shape)
            data_np_imag = np.random.randn(*self.shape)
            data_np = data_np_real + data_np_imag * 1j
            data_np = data_np.astype(init_dtype)
        else:
            data_np = np.random.randint(-100, 100, size=self.shape).astype(
                init_dtype
            )

        tensor = paddle.to_tensor(data_np, dtype=init_dtype)

        # Check if conversion method exists
        self.assertTrue(
            hasattr(tensor, method_name),
            f"Tensor should have method '{method_name}'",
        )
        # Perform dtype conversion
        converted_tensor = getattr(tensor, method_name)()

        # Check the dtype after conversion
        expected_dtype = self._get_paddle_dtype(target_dtype)
        self.assertEqual(
            converted_tensor.dtype,
            expected_dtype,
            f"Expected dtype {expected_dtype}, but got {converted_tensor.dtype} for method '{method_name}'",
        )

        # Check that the shape remains unchanged
        self.assertEqual(
            tensor.shape,
            converted_tensor.shape,
            f"Shape should remain unchanged after {method_name} conversion",
        )

        if target_dtype.endswith('float16'):
            rtol = 1e-3
            atol = 1e-3
        else:
            rtol = 1e-7
            atol = 0

        # Check the value after conversion
        np.testing.assert_allclose(
            converted_tensor.numpy(),
            data_np.astype(target_dtype),
            rtol=rtol,
            atol=atol,
            err_msg=f"Value mismatch after {method_name} conversion",
        )

    def test_method_chaining(self):
        """Test method chaining for dtype conversions."""
        tensor = paddle.to_tensor([1.5, 2.5, 3.5], dtype='float32')

        # float32 -> int32 -> float64 -> int64
        result = tensor.int32().float64().int64()
        self.assertEqual(result.dtype, paddle.int64)

    def test_pir_all_dtype_conversions(self):
        """Test all dtype conversion methods for pir.Value in static graph."""
        paddle.enable_static()
        startup_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            for (
                method_name,
                target_dtype,
            ) in self._supported_dtype_conversions.items():

                if target_dtype == 'bfloat16':
                    continue
                for init_dtype in self._total_init_dtype:
                    if (
                        self._device.startswith('xpu')
                        and target_dtype == 'complex128'
                    ):
                        self.skipTest(
                            "Skipping complex conversion tests on XPU"
                        )
                    with self.subTest(
                        pir_method=method_name,
                        pir_init_dtype=init_dtype,
                        pir_target_dtype=target_dtype,
                    ):
                        self._pir_single_dtype_conversion(
                            method_name, init_dtype, target_dtype
                        )

    def _pir_single_dtype_conversion(
        self, method_name, init_dtype, target_dtype
    ):

        # Create static graph input
        x = paddle.static.data(name="x", shape=self.shape, dtype=init_dtype)
        # Check if the method exists
        self.assertTrue(
            hasattr(x, method_name),
            f"pir.Value should have method '{method_name}'",
        )
        # Perform dtype conversion
        converted = getattr(x, method_name)()
        # Check the dtype
        expected_dtype = self._get_paddle_dtype(target_dtype)
        self.assertEqual(
            converted.dtype,
            expected_dtype,
            f"Expected pir.Value dtype {expected_dtype}, but got {converted.dtype} for method '{method_name}'",
        )
        # Check the shape
        self.assertEqual(
            tuple(x.shape),
            tuple(converted.shape),
            f"pir.Value shape should remain unchanged after {method_name} conversion",
        )


if __name__ == '__main__':
    unittest.main()
