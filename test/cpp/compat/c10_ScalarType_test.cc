// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/tensor.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

TEST(TensorBaseTest, TypeCheckingAPIs) {
  // Test is_complex()
  at::TensorBase complex64_tensor = at::ones({2, 3}, at::kComplexFloat);
  at::TensorBase complex128_tensor = at::ones({2, 3}, at::kComplexDouble);
  at::TensorBase float_tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase int_tensor = at::ones({2, 3}, at::kInt);

  ASSERT_TRUE(complex64_tensor.is_complex());
  ASSERT_TRUE(complex128_tensor.is_complex());
  ASSERT_FALSE(float_tensor.is_complex());
  ASSERT_FALSE(int_tensor.is_complex());

  // Test is_floating_point()
  at::TensorBase float16_tensor = at::ones({2, 3}, at::kHalf);
  at::TensorBase bfloat16_tensor = at::ones({2, 3}, at::kBFloat16);
  at::TensorBase float32_tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase float64_tensor = at::ones({2, 3}, at::kDouble);
  at::TensorBase float8_e5m2_tensor = at::ones({2, 3}, at::kFloat8_e5m2);
  at::TensorBase float8_e4m3fn_tensor = at::ones({2, 3}, at::kFloat8_e4m3fn);
  at::TensorBase bool_tensor = at::ones({2, 3}, at::kBool);

  ASSERT_TRUE(float16_tensor.is_floating_point());
  ASSERT_TRUE(bfloat16_tensor.is_floating_point());
  ASSERT_TRUE(float32_tensor.is_floating_point());
  ASSERT_TRUE(float64_tensor.is_floating_point());
  ASSERT_TRUE(float8_e5m2_tensor.is_floating_point());
  ASSERT_TRUE(float8_e4m3fn_tensor.is_floating_point());
  ASSERT_FALSE(int_tensor.is_floating_point());
  ASSERT_FALSE(bool_tensor.is_floating_point());
  ASSERT_FALSE(complex64_tensor.is_floating_point());

  // Test is_signed()
  at::TensorBase int8_tensor = at::ones({2, 3}, at::kChar);
  at::TensorBase int16_tensor = at::ones({2, 3}, at::kShort);
  at::TensorBase int32_tensor = at::ones({2, 3}, at::kInt);
  at::TensorBase int64_tensor = at::ones({2, 3}, at::kLong);
  at::TensorBase uint8_tensor = at::ones({2, 3}, at::kByte);

  // Signed integer types
  ASSERT_TRUE(int8_tensor.is_signed());
  ASSERT_TRUE(int16_tensor.is_signed());
  ASSERT_TRUE(int32_tensor.is_signed());
  ASSERT_TRUE(int64_tensor.is_signed());

  // Floating point types are signed
  ASSERT_TRUE(float16_tensor.is_signed());
  ASSERT_TRUE(bfloat16_tensor.is_signed());
  ASSERT_TRUE(float32_tensor.is_signed());
  ASSERT_TRUE(float64_tensor.is_signed());
  ASSERT_TRUE(float8_e5m2_tensor.is_signed());
  ASSERT_TRUE(float8_e4m3fn_tensor.is_signed());

  // Complex types are signed
  ASSERT_TRUE(complex64_tensor.is_signed());
  ASSERT_TRUE(complex128_tensor.is_signed());

  // Unsigned types
  ASSERT_FALSE(uint8_tensor.is_signed());
  ASSERT_FALSE(bool_tensor.is_signed());
}
