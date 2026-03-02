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
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#include <limits>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

TEST(TestAll, AllNoDim) {
  // Test all() without arguments - check all elements in tensor
  at::Tensor tensor = at::ones({3}, at::kBool);
  tensor[1] = false;
  at::Tensor result = tensor.all();

  ASSERT_EQ(result.numel(), 1);
  ASSERT_EQ(result.item<bool>(), false);

  // Test with all true values
  at::Tensor tensor_all_true = at::ones({3}, at::kBool);
  at::Tensor result_all_true = tensor_all_true.all();
  ASSERT_EQ(result_all_true.item<bool>(), true);
}

TEST(TestAll, AllWithDim) {
  // Test all(dim) - check along specific dimension
  at::Tensor tensor = at::ones({2, 2}, at::kBool);
  tensor[1][0] = false;

  // All along dimension 0
  at::Tensor result_dim0 = tensor.all(0);
  ASSERT_EQ(result_dim0.sizes(), c10::IntArrayRef({2}));
  ASSERT_EQ(result_dim0.data_ptr<bool>()[0], false);  // column 0 has false
  ASSERT_EQ(result_dim0.data_ptr<bool>()[1], true);   // column 1 has all true

  // All along dimension 1
  at::Tensor result_dim1 = tensor.all(1);
  ASSERT_EQ(result_dim1.sizes(), c10::IntArrayRef({2}));
  ASSERT_EQ(result_dim1.data_ptr<bool>()[0], true);   // row 0 has all true
  ASSERT_EQ(result_dim1.data_ptr<bool>()[1], false);  // row 1 has false
}

TEST(TestAll, AllWithDimKeepdim) {
  // Test all(dim, keepdim) - keep the dimension
  at::Tensor tensor = at::ones({2, 2}, at::kBool);

  at::Tensor result = tensor.all(0, true);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({1, 2}));
}

TEST(TestAll, AllWithOptionalDim) {
  // Test all(OptionalIntArrayRef dim, keepdim)
  at::Tensor tensor = at::ones({2, 2}, at::kBool);

  // With specific dimensions
  at::Tensor result = tensor.all(c10::IntArrayRef({0}), false);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2}));
}

TEST(TestAll, StandaloneFunction) {
  // Test at::all() standalone function
  at::Tensor tensor = at::ones({3}, at::kBool);
  tensor[2] = false;
  at::Tensor result = at::all(tensor);

  ASSERT_EQ(result.item<bool>(), false);
}

TEST(TestAllclose, AllcloseBasic) {
  // Test allclose - basic equal tensors
  at::Tensor tensor1 = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor tensor2 = at::arange(6, at::kFloat).reshape({2, 3});

  bool result = tensor1.allclose(tensor2);
  ASSERT_EQ(result, true);
}

TEST(TestAllclose, AllcloseNotEqual) {
  // Test allclose - tensors that are not close
  at::Tensor tensor1 = at::arange(1, 4, at::TensorOptions().dtype(at::kFloat));
  at::Tensor tensor2 = tensor1.clone();
  tensor2[2] = 4.0f;

  bool result = tensor1.allclose(tensor2);
  ASSERT_EQ(result, false);
}

TEST(TestAllclose, StandaloneFunction) {
  // Test at::allclose() standalone function
  at::Tensor tensor1 = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor tensor2 = at::arange(6, at::kFloat).reshape({2, 3});

  bool result = at::allclose(tensor1, tensor2);
  ASSERT_EQ(result, true);
}
