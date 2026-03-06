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

import numpy as np

import paddle

print("=" * 60)
print("Testing paddle.cumsum with bool and int types")
print("=" * 60)

# Test 1: bool type - basic
print("\n--- Test 1: bool cumsum (flatten) ---")
data_bool = paddle.to_tensor([True, False, True, True, False, True])
y = paddle.cumsum(data_bool)
expected = np.cumsum(data_bool.numpy().astype(np.int64))
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print(f"  input:    {data_bool.numpy()}")
print(f"  output:   {y.numpy()}")
print(f"  expected: {expected}")
print(f"  dtype:    {y.dtype}")
print("  PASSED")

# Test 2: bool type - with axis=0
print("\n--- Test 2: bool cumsum (axis=0) ---")
data_bool_2d = paddle.to_tensor([[True, False, True], [True, True, False]])
y = paddle.cumsum(data_bool_2d, axis=0)
expected = np.cumsum(data_bool_2d.numpy().astype(np.int64), axis=0)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print(f"  output:\n{y.numpy()}")
print("  PASSED")

# Test 3: bool type - with axis=1
print("\n--- Test 3: bool cumsum (axis=1) ---")
y = paddle.cumsum(data_bool_2d, axis=1)
expected = np.cumsum(data_bool_2d.numpy().astype(np.int64), axis=1)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print(f"  output:\n{y.numpy()}")
print("  PASSED")

# Test 4: bool type - axis=-1
print("\n--- Test 4: bool cumsum (axis=-1) ---")
y = paddle.cumsum(data_bool_2d, axis=-1)
expected = np.cumsum(data_bool_2d.numpy().astype(np.int64), axis=-1)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print("  PASSED")

# Test 5: bool all-True
print("\n--- Test 5: bool all-True cumsum ---")
data_all_true = paddle.to_tensor([True, True, True, True])
y = paddle.cumsum(data_all_true)
expected = np.array([1, 2, 3, 4], dtype=np.int64)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print(f"  output: {y.numpy()}")
print("  PASSED")

# Test 6: bool all-False
print("\n--- Test 6: bool all-False cumsum ---")
data_all_false = paddle.to_tensor([False, False, False])
y = paddle.cumsum(data_all_false)
expected = np.array([0, 0, 0], dtype=np.int64)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print(f"  output: {y.numpy()}")
print("  PASSED")

# Test 7: bool 3D tensor
print("\n--- Test 7: bool 3D cumsum ---")
data_3d = paddle.to_tensor(
    [[[True, False], [True, True]], [[False, True], [True, False]]]
)
y = paddle.cumsum(data_3d, axis=2)
expected = np.cumsum(data_3d.numpy().astype(np.int64), axis=2)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print(f"  output:\n{y.numpy()}")
print("  PASSED")

# Test 8: bool 3D axis=0
print("\n--- Test 8: bool 3D cumsum (axis=0) ---")
y = paddle.cumsum(data_3d, axis=0)
expected = np.cumsum(data_3d.numpy().astype(np.int64), axis=0)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print("  PASSED")

# Test 9: bool 3D axis=1
print("\n--- Test 9: bool 3D cumsum (axis=1) ---")
y = paddle.cumsum(data_3d, axis=1)
expected = np.cumsum(data_3d.numpy().astype(np.int64), axis=1)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print("  PASSED")

# Test 10: float32 type - should stay float32 (not affected)
print("\n--- Test 10: float32 cumsum ---")
data_float = paddle.ones([3, 4], dtype='float32')
y = paddle.cumsum(data_float, axis=0)
expected = np.cumsum(data_float.numpy(), axis=0)
np.testing.assert_allclose(y.numpy(), expected)
assert y.dtype == paddle.float32, f"Expected float32, got {y.dtype}"
print("  PASSED")

# Test 11: int64 type - should stay int64
print("\n--- Test 11: int64 cumsum ---")
data_int64 = paddle.arange(12, dtype='int64').reshape((3, 4))
y = paddle.cumsum(data_int64, axis=1)
expected = np.cumsum(data_int64.numpy(), axis=1)
np.testing.assert_array_equal(y.numpy(), expected)
assert y.dtype == paddle.int64, f"Expected int64, got {y.dtype}"
print(f"  output:\n{y.numpy()}")
print("  PASSED")

# Test 12: explicit dtype override still works
print("\n--- Test 12: explicit dtype=float64 ---")
data_bool_cast = paddle.to_tensor([True, False, True])
y = paddle.cumsum(data_bool_cast, dtype='float64')
expected = np.cumsum(data_bool_cast.numpy().astype(np.float64))
np.testing.assert_allclose(y.numpy(), expected)
assert y.dtype == paddle.float64, f"Expected float64, got {y.dtype}"
print(f"  output: {y.numpy()}")
print("  PASSED")

# Note: int8/int16/int32/uint8 -> int64 promotion tests require
# recompilation of InferMeta C++ code. After recompilation, the
# following tests should also pass:
# - int32 cumsum -> int64 output
# - int8 cumsum -> int64 output
# - int16 cumsum -> int64 output
# - uint8 cumsum -> int64 output

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
