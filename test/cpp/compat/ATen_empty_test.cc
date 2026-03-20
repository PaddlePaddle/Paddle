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
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ======================== at::empty basic tests ========================

TEST(ATenEmptyTest, BasicShape) {
  at::Tensor t = at::empty({3, 4});
  ASSERT_EQ(t.sizes()[0], 3);
  ASSERT_EQ(t.sizes()[1], 4);
}

TEST(ATenEmptyTest, DtypeFloat) {
  at::Tensor t = at::empty({2, 2}, at::TensorOptions().dtype(at::kFloat));
  ASSERT_EQ(t.scalar_type(), at::kFloat);
}

TEST(ATenEmptyTest, DtypeDouble) {
  at::Tensor t = at::empty({4}, at::TensorOptions().dtype(at::kDouble));
  ASSERT_EQ(t.scalar_type(), at::kDouble);
}

TEST(ATenEmptyTest, ExplicitArgsCpu) {
  // 6-argument overload: dtype, layout, device, pin_memory, memory_format
  at::Tensor t = at::empty(
      {2, 3}, at::kFloat, at::kStrided, at::kCPU, false, std::nullopt);
  ASSERT_EQ(t.sizes()[0], 2);
  ASSERT_EQ(t.sizes()[1], 3);
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  ASSERT_FALSE(t.is_pinned());
}

// ======================== pin_memory tests ========================

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

// TensorOptions overload: pin_memory via options
TEST(ATenEmptyTest, PinMemoryViaTensorOptions) {
  at::TensorOptions opts =
      at::TensorOptions().dtype(at::kFloat).pinned_memory(true);
  at::Tensor t = at::empty({4, 4}, opts);
  ASSERT_TRUE(t.is_pinned())
      << "Expected pinned memory tensor when TensorOptions.pinned_memory=true";
}

// 6-argument overload: pin_memory = true (must use CPU device)
TEST(ATenEmptyTest, PinMemoryViaExplicitArgs) {
  at::Tensor t =
      at::empty({8}, at::kFloat, at::kStrided, at::kCPU, true, std::nullopt);
  ASSERT_TRUE(t.is_pinned())
      << "Expected pinned memory tensor when pin_memory=true with CPU device";
}

// pin_memory = false must NOT produce a pinned tensor
TEST(ATenEmptyTest, NoPinMemoryViaExplicitArgs) {
  at::Tensor t =
      at::empty({8}, at::kFloat, at::kStrided, at::kCUDA, false, std::nullopt);
  ASSERT_FALSE(t.is_pinned())
      << "Expected non-pinned tensor when pin_memory=false";
}

// Pinned tensor lives in pinned (host) memory, not on the GPU device itself
TEST(ATenEmptyTest, PinnedTensorIsNotCuda) {
  at::TensorOptions opts =
      at::TensorOptions().dtype(at::kFloat).pinned_memory(true);
  at::Tensor t = at::empty({16}, opts);
  ASSERT_TRUE(t.is_pinned());
  ASSERT_FALSE(t.is_cuda())
      << "Pinned tensor should reside in host pinned memory, not on device";
}

// Data pointer of a pinned tensor must be non-null
TEST(ATenEmptyTest, PinnedTensorDataPtrNonNull) {
  at::TensorOptions opts =
      at::TensorOptions().dtype(at::kFloat).pinned_memory(true);
  at::Tensor t = at::empty({32}, opts);
  ASSERT_TRUE(t.is_pinned());
  ASSERT_NE(t.data_ptr(), nullptr);
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
