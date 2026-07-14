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

// Define PADDLE_WITH_CUSTOM_KERNEL before any Paddle header so that
// dense_tensor.inl is excluded from this translation unit. That hides
// phi::DenseTensor::ResetHolder from the compiler, which makes the SFINAE
// template overload of TensorBase::MaybeResetHolder drop out and forces the
// fallback overload to be selected -- the same path external custom-kernel
// extensions take. Guards PR #78826's fix to that fallback.
#define PADDLE_WITH_CUSTOM_KERNEL

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/resize.h>
#include <c10/core/ScalarType.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"

TEST(ATenResizeCustomKernel, ResizeGrowsStorageInFallbackPath) {
  at::Tensor t = at::ones({2}, at::kInt);
  ASSERT_EQ(t.numel(), 2);
  ASSERT_EQ(t.storage().nbytes(), 8u);

  t.resize_({4});

  ASSERT_EQ(t.numel(), 4);
  // Without PR #78826 the storage().nbytes() stays at 8 because
  // SyncStorageFromTensor rebuilds StorageImpl from the stale holder.
  ASSERT_GE(t.storage().nbytes(), 16u);
}

// Covers the numel() == 0 branch in MaybeResetHolder fallback.
// at::empty({0}) creates a tensor whose initial holder is a plain
// Allocation (not yet synced to StorageHolderView). During TensorBase
// construction, SyncStorageFromTensor sees holder != compat_holder and
// calls MaybeResetHolder with dense->numel() == 0, forcing the
// offset-reset branch.
TEST(ATenResizeCustomKernel, EmptyTensorOffsetResetInFallbackPath) {
  at::Tensor t = at::empty({0}, at::kInt);
  ASSERT_EQ(t.numel(), 0);

  // storage() triggers SyncStorageFromTensor -> MaybeResetHolder
  // with dense->numel() == 0, covering the offset-reset branch.
  auto s = t.storage();
  ASSERT_TRUE(s.valid());
  ASSERT_EQ(t.numel(), 0);
}
