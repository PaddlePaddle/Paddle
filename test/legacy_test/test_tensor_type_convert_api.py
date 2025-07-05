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
        'bfloat16': 'bfloat16',
        'float32': 'float32',
        'float64': 'float64',
        # int
        'int8': 'int8',
        'int16': 'int16',
        'int32': 'int32',
        'int64': 'int64',
        # other
        'bool': 'bool',
        'complex64': 'complex64',
        'complex128': 'complex128',
    }

    def setUp(self):
        """Set up test data for different types."""
        self.test_values = {
            'float': np.array([1.5, 2.5, 3.5]),
            'int': np.array([1, 2, 3]),
            'bool': np.array([True, False, True]),
            'complex': np.array([1 + 2j, 3 + 4j, 5 + 6j]),
        }

    def _get_paddle_dtype(self, dtype_str):
        """Get the Paddle dtype constant by string name."""
        return getattr(paddle, dtype_str)

    def _get_appropriate_test_data(self, target_dtype):
        """Select appropriate test data according to target dtype."""
        if target_dtype in ['complex64', 'complex128']:
            return self.test_values['complex']
        elif target_dtype == 'bool':
            return self.test_values['bool']
        elif target_dtype.startswith(('int', 'long')):
            return self.test_values['int']
        else:  # float types
            return self.test_values['float']

    def test_all_dtype_conversions(self):
        """Test all dtype conversion methods."""
        for (
            method_name,
            target_dtype,
        ) in self._supported_dtype_conversions.items():
            with self.subTest(method=method_name, target_dtype=target_dtype):
                self._test_single_dtype_conversion(method_name, target_dtype)

    def _test_single_dtype_conversion(self, method_name, target_dtype):
        """Test a single dtype conversion method."""
        # Select appropriate test data
        test_data = self._get_appropriate_test_data(target_dtype)

        # Create initial tensor (use float32 unless special type)
        if target_dtype in ['complex64', 'complex128']:
            initial_dtype = 'complex64'
        elif target_dtype == 'bool':
            initial_dtype = 'bool'
        else:
            initial_dtype = 'float32'

        tensor = paddle.to_tensor(test_data, dtype=initial_dtype)

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

    def test_float_to_int_conversion(self):
        """Test float to int conversion."""
        float_tensor = paddle.to_tensor([1.7, 2.3, 3.9], dtype='float32')
        int_tensor = float_tensor.int32()

        self.assertEqual(int_tensor.dtype, paddle.int32)

    def test_int_to_float_conversion(self):
        """Test int to float conversion."""
        int_tensor = paddle.to_tensor([1, 2, 3], dtype='int32')
        float_tensor = int_tensor.float32()

        self.assertEqual(float_tensor.dtype, paddle.float32)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(float_tensor.numpy(), expected)

    def test_complex_conversions(self):
        """Test complex dtype conversions."""
        # Create a complex tensor
        complex_data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        tensor = paddle.to_tensor(complex_data, dtype='complex64')

        # Test complex64 to complex128
        complex128_tensor = tensor.complex128()
        self.assertEqual(complex128_tensor.dtype, paddle.complex128)

        # Test complex128 to complex64
        complex64_tensor = complex128_tensor.complex64()
        self.assertEqual(complex64_tensor.dtype, paddle.complex64)

    def test_bool_conversions(self):
        """Test bool dtype conversions."""
        # From numeric to bool
        numeric_tensor = paddle.to_tensor([0, 1, 2, -1], dtype='int32')
        bool_tensor = numeric_tensor.bool()
        self.assertEqual(bool_tensor.dtype, paddle.bool)

        # From bool to numeric
        bool_data = paddle.to_tensor([True, False, True], dtype='bool')
        int_tensor = bool_data.int32()
        self.assertEqual(int_tensor.dtype, paddle.int32)
        expected = np.array([1, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(int_tensor.numpy(), expected)

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
                with self.subTest(
                    pir_method=method_name, pir_target_dtype=target_dtype
                ):
                    self._pir_single_dtype_conversion(method_name, target_dtype)

    def _pir_single_dtype_conversion(self, method_name, target_dtype):
        # Select appropriate test data
        test_data = self._get_appropriate_test_data(target_dtype)
        shape = test_data.shape
        dtype = 'float32'
        if target_dtype in ['complex64', 'complex128']:
            dtype = 'complex64'
        elif target_dtype == 'bool':
            dtype = 'bool'
        # Create static graph input
        x = paddle.static.data(name="x", shape=shape, dtype=dtype)
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
