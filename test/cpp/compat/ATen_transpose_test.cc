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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

// ============================================================
// Tests for at::Tensor::transpose_(int64_t dim0, int64_t dim1)
// (in-place variant; out-of-place transpose is a separate path)
// ============================================================

TEST(TensorTransposeInplaceTest, Transpose2D_SwapDims) {
  // transpose_(0, 1) on a 2D tensor: shape becomes transposed in-place
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  void* original_ptr = t.data_ptr();
  at::Tensor& ref = t.transpose_(0, 1);

  ASSERT_EQ(t.dim(), 2);
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({4, 3}));
  // Returns *this by reference
  ASSERT_EQ(&ref, &t);
  ASSERT_EQ(t.data_ptr(), original_ptr);
}

TEST(TensorTransposeInplaceTest, Transpose3D_SwapFirstTwo) {
  // transpose_(0, 1) on a 3D tensor swaps first two axes in-place
  at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
  void* original_ptr = t.data_ptr();
  t.transpose_(0, 1);

  ASSERT_EQ(t.dim(), 3);
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({3, 2, 4}));
  ASSERT_EQ(t.data_ptr(), original_ptr);
}

TEST(TensorTransposeInplaceTest, Transpose3D_SwapLastTwo) {
  // transpose_(1, 2) on a 3D tensor swaps last two axes in-place
  at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
  void* original_ptr = t.data_ptr();
  t.transpose_(1, 2);

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({2, 4, 3}));
  ASSERT_EQ(t.data_ptr(), original_ptr);
}

TEST(TensorTransposeInplaceTest, TransposeInplace_PreservesValues) {
  // Verify values are correctly accessed after in-place transpose
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  // t = [[0,1,2],[3,4,5]]
  t.transpose_(0, 1);
  // After t: shape {3,2}, layout: [[0,3],[1,4],[2,5]]

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({3, 2}));
  ASSERT_FLOAT_EQ(t[0][0].item<float>(), 0.0f);
  ASSERT_FLOAT_EQ(t[0][1].item<float>(), 3.0f);
  ASSERT_FLOAT_EQ(t[2][0].item<float>(), 2.0f);
  ASSERT_FLOAT_EQ(t[2][1].item<float>(), 5.0f);
}

TEST(TensorTransposeInplaceTest, TransposeInplace_SameDim_NoOp) {
  // transpose_(i, i) is a no-op; shape and data pointer are unchanged
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  void* original_ptr = t.data_ptr();
  t.transpose_(0, 0);

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({3, 4}));
  ASSERT_EQ(t.data_ptr(), original_ptr);
}

TEST(TensorTransposeInplaceTest,
     TransposeInplace_DoubleTranspose_RestoresShape) {
  // Two consecutive in-place transposes on same dims restore original shape
  at::Tensor t = at::ones({5, 7}, at::kFloat);
  t.transpose_(0, 1);
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({7, 5}));
  t.transpose_(0, 1);
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({5, 7}));
}
