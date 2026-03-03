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

TEST(TestAll, AllNoDimAllFalse) {
  // Test all() on tensor with all false values
  at::Tensor tensor = at::zeros({4}, at::kBool);
  at::Tensor result = tensor.all();
  ASSERT_EQ(result.numel(), 1);
  ASSERT_EQ(result.item<bool>(), false);
}

TEST(TestAll, AllNoDimSingleElement) {
  // Test all() on single-element tensor
  at::Tensor tensor_true = at::ones({1}, at::kBool);
  ASSERT_EQ(tensor_true.all().item<bool>(), true);

  at::Tensor tensor_false = at::zeros({1}, at::kBool);
  ASSERT_EQ(tensor_false.all().item<bool>(), false);
}

TEST(TestAll, AllWithNegativeDim) {
  // Test all(dim) with negative dimension index
  at::Tensor tensor = at::ones({2, 3}, at::kBool);
  tensor[0][1] = false;
  at::Tensor result = tensor.all(-1);  // equivalent to dim=1
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2}));
  ASSERT_EQ(result.data_ptr<bool>()[0], false);  // row 0 has a false
  ASSERT_EQ(result.data_ptr<bool>()[1], true);   // row 1 all true
}

TEST(TestAll, AllWithDimKeepdimTrue) {
  // Test all(dim, keepdim=true) with different dims
  at::Tensor tensor = at::ones({2, 3}, at::kBool);
  tensor[1][0] = false;

  at::Tensor result_dim0 = tensor.all(0, true);
  ASSERT_EQ(result_dim0.sizes(), c10::IntArrayRef({1, 3}));
  ASSERT_EQ(result_dim0.data_ptr<bool>()[0], false);  // col 0 has false
  ASSERT_EQ(result_dim0.data_ptr<bool>()[1], true);
  ASSERT_EQ(result_dim0.data_ptr<bool>()[2], true);

  at::Tensor result_dim1 = tensor.all(1, true);
  ASSERT_EQ(result_dim1.sizes(), c10::IntArrayRef({2, 1}));
  ASSERT_EQ(result_dim1.data_ptr<bool>()[0], true);   // row 0 all true
  ASSERT_EQ(result_dim1.data_ptr<bool>()[1], false);  // row 1 has false
}

TEST(TestAll, AllWithOptionalDimNullopt) {
  // Test all(OptionalIntArrayRef) with nullopt - reduces all dimensions
  at::Tensor tensor = at::ones({2, 3}, at::kBool);
  at::OptionalIntArrayRef dim = std::nullopt;
  at::Tensor result = at::all(tensor, dim, false);
  ASSERT_EQ(result.numel(), 1);
  ASSERT_EQ(result.item<bool>(), true);
}

TEST(TestAll, AllWithOptionalDimNulloptHasFalse) {
  // Test all(OptionalIntArrayRef nullopt) when tensor contains false
  at::Tensor tensor = at::ones({2, 3}, at::kBool);
  tensor[1][2] = false;
  at::OptionalIntArrayRef dim = std::nullopt;
  at::Tensor result = at::all(tensor, dim, false);
  ASSERT_EQ(result.numel(), 1);
  ASSERT_EQ(result.item<bool>(), false);
}

TEST(TestAll, AllWithOptionalDimKeepdim) {
  // Test all(OptionalIntArrayRef, keepdim=true)
  at::Tensor tensor = at::ones({2, 3}, at::kBool);
  at::Tensor result = at::all(tensor, c10::IntArrayRef({0}), true);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({1, 3}));
}

TEST(TestAll, AllWithOptionalMultipleDims) {
  // Test all(OptionalIntArrayRef) with multiple dimensions
  at::Tensor tensor = at::ones({2, 3, 4}, at::kBool);
  at::Tensor result = at::all(tensor, c10::IntArrayRef({0, 2}), false);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3}));
  // All elements are true, so result should be all true
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(result.data_ptr<bool>()[i], true);
  }
}

TEST(TestAll, MemberAllWithOptionalNullopt) {
  // Test member function Tensor::all(OptionalIntArrayRef, keepdim) with nullopt
  at::Tensor tensor = at::ones({3, 4}, at::kBool);
  at::OptionalIntArrayRef dim = std::nullopt;
  at::Tensor result = tensor.all(dim, false);
  ASSERT_EQ(result.numel(), 1);
  ASSERT_EQ(result.item<bool>(), true);
}

TEST(TestAll, MemberAllWithOptionalNulloptKeepdim) {
  // Test member function Tensor::all(nullopt, keepdim=true)
  at::Tensor tensor = at::ones({2, 3}, at::kBool);
  at::OptionalIntArrayRef dim = std::nullopt;
  at::Tensor result = tensor.all(dim, true);
  ASSERT_EQ(result.numel(), 1);
  ASSERT_EQ(result.item<bool>(), true);
}

TEST(TestAll, StandaloneFunction) {
  // Test at::all() standalone function
  at::Tensor tensor = at::ones({3}, at::kBool);
  tensor[2] = false;
  at::Tensor result = at::all(tensor);

  ASSERT_EQ(result.item<bool>(), false);
}

TEST(TestAll, StandaloneFunctionWithDim) {
  // Test at::all(tensor, dim, keepdim)
  at::Tensor tensor = at::ones({2, 3}, at::kBool);
  tensor[0][0] = false;

  at::Tensor result = at::all(tensor, 0, false);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3}));
  ASSERT_EQ(result.data_ptr<bool>()[0], false);
  ASSERT_EQ(result.data_ptr<bool>()[1], true);
  ASSERT_EQ(result.data_ptr<bool>()[2], true);

  at::Tensor result_kd = at::all(tensor, 0, true);
  ASSERT_EQ(result_kd.sizes(), c10::IntArrayRef({1, 3}));
}

TEST(TestAll, AllWith3DTensor) {
  // Test all on a 3D tensor to exercise more paths
  at::Tensor tensor = at::ones({2, 2, 2}, at::kBool);
  tensor[0][0][0] = false;

  at::Tensor result_all = tensor.all();
  ASSERT_EQ(result_all.item<bool>(), false);

  at::Tensor result_dim0 = tensor.all(0, false);
  ASSERT_EQ(result_dim0.sizes(), c10::IntArrayRef({2, 2}));

  at::Tensor result_dim2 = tensor.all(2, true);
  ASSERT_EQ(result_dim2.sizes(), c10::IntArrayRef({2, 2, 1}));
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

TEST(TestAllclose, AllcloseWithCustomRtol) {
  // Test allclose with custom relative tolerance
  at::Tensor tensor1 = at::ones({3}, at::kFloat);
  at::Tensor tensor2 = at::ones({3}, at::kFloat);
  tensor2[0] = 1.05f;  // 5% difference

  // With default rtol=1e-05, should fail
  bool result_default = at::allclose(tensor1, tensor2);
  ASSERT_EQ(result_default, false);

  // With rtol=0.1 (10%), 5% difference should pass
  bool result_large_rtol = at::allclose(tensor1, tensor2, 0.1, 1e-08, false);
  ASSERT_EQ(result_large_rtol, true);
}

TEST(TestAllclose, AllcloseWithCustomAtol) {
  // Test allclose with custom absolute tolerance
  at::Tensor tensor1 = at::zeros({3}, at::kFloat);
  at::Tensor tensor2 = at::zeros({3}, at::kFloat);
  tensor2[1] = 0.05f;

  // With default atol=1e-08, should fail
  bool result_default = at::allclose(tensor1, tensor2);
  ASSERT_EQ(result_default, false);

  // With atol=0.1, should pass
  bool result_large_atol = at::allclose(tensor1, tensor2, 1e-05, 0.1, false);
  ASSERT_EQ(result_large_atol, true);
}

TEST(TestAllclose, AllcloseMemberWithAllParams) {
  // Test Tensor::allclose member function with all explicit parameters
  at::Tensor tensor1 = at::ones({2, 2}, at::kFloat);
  at::Tensor tensor2 = at::ones({2, 2}, at::kFloat);

  bool result = tensor1.allclose(tensor2, 1e-05, 1e-08, false);
  ASSERT_EQ(result, true);
}

TEST(TestAllclose, AllcloseMemberNotClose) {
  // Test Tensor::allclose member function returns false when not close
  at::Tensor tensor1 = at::ones({2, 3}, at::kFloat);
  at::Tensor tensor2 = at::ones({2, 3}, at::kFloat);
  tensor2[0][0] = 100.0f;

  bool result = tensor1.allclose(tensor2, 1e-05, 1e-08, false);
  ASSERT_EQ(result, false);
}

TEST(TestAllclose, AllcloseMemberWithCustomTolerance) {
  // Test Tensor::allclose member function with custom rtol and atol
  at::Tensor tensor1 = at::ones({4}, at::kFloat);
  at::Tensor tensor2 = at::ones({4}, at::kFloat);
  tensor2[3] = 1.001f;  // small relative difference

  // Default tolerance should fail
  ASSERT_EQ(tensor1.allclose(tensor2), false);

  // Custom rtol=0.01 (1%) should pass
  ASSERT_EQ(tensor1.allclose(tensor2, 0.01, 1e-08, false), true);
}

TEST(TestAllclose, AllcloseExactZeros) {
  // Test allclose with exact zero tensors
  at::Tensor tensor1 = at::zeros({5}, at::kFloat);
  at::Tensor tensor2 = at::zeros({5}, at::kFloat);

  bool result = at::allclose(tensor1, tensor2);
  ASSERT_EQ(result, true);

  bool result_member = tensor1.allclose(tensor2);
  ASSERT_EQ(result_member, true);
}

TEST(TestAllclose, AllcloseHighDim) {
  // Test allclose with higher dimensional tensors
  at::Tensor tensor1 = at::arange(24, at::kFloat).reshape({2, 3, 4});
  at::Tensor tensor2 = at::arange(24, at::kFloat).reshape({2, 3, 4});

  bool result = at::allclose(tensor1, tensor2);
  ASSERT_EQ(result, true);

  bool result_member = tensor1.allclose(tensor2, 1e-05, 1e-08, false);
  ASSERT_EQ(result_member, true);
}
