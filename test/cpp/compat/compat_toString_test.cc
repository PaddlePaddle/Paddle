// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

TEST(TensorBaseTest, ToStringAPI) {
  // Test toString() API
  at::TensorBase cpu_float_tensor = at::ones({2, 3}, at::kFloat);
  std::string cpu_float_str = cpu_float_tensor.toString();
  ASSERT_EQ(cpu_float_str, "CPUFloatType");

  at::TensorBase cpu_double_tensor = at::ones({2, 3}, at::kDouble);
  std::string cpu_double_str = cpu_double_tensor.toString();
  ASSERT_EQ(cpu_double_str, "CPUDoubleType");

  at::TensorBase cpu_int_tensor = at::ones({2, 3}, at::kInt);
  std::string cpu_int_str = cpu_int_tensor.toString();
  ASSERT_EQ(cpu_int_str, "CPUIntType");

  at::TensorBase cpu_long_tensor = at::ones({2, 3}, at::kLong);
  std::string cpu_long_str = cpu_long_tensor.toString();
  ASSERT_EQ(cpu_long_str, "CPULongType");

  at::TensorBase cpu_bool_tensor = at::ones({2, 3}, at::kBool);
  std::string cpu_bool_str = cpu_bool_tensor.toString();
  ASSERT_EQ(cpu_bool_str, "CPUBoolType");

  // Test undefined tensor
  at::TensorBase undefined_tensor;
  std::string undefined_str = undefined_tensor.toString();
  ASSERT_EQ(undefined_str, "UndefinedType");

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Test CUDA tensor if available
  at::TensorBase cuda_float_tensor = at::ones(
      {2, 3},
      at::TensorOptions().dtype(at::kFloat).device(at::Device(at::kCUDA, 0)));
  std::string cuda_float_str = cuda_float_tensor.toString();
  ASSERT_EQ(cuda_float_str, "CUDAFloatType");

  at::TensorBase cuda_double_tensor = at::ones(
      {2, 3},
      at::TensorOptions().dtype(at::kDouble).device(at::Device(at::kCUDA, 0)));
  std::string cuda_double_str = cuda_double_tensor.toString();
  ASSERT_EQ(cuda_double_str, "CUDADoubleType");
#endif
}
