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

// Verify that including both headers in the same translation unit compiles
// cleanly (no ODR violations) and that the canonical PhiloxCudaState
// definition is consistent across both include paths.
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>

#include <gtest/gtest.h>

// offset_intragraph_ must be uint64_t to match PyTorch upstream.
static_assert(std::is_same_v<decltype(at::PhiloxCudaState{}.offset_intragraph_),
                             uint64_t>,
              "PhiloxCudaState::offset_intragraph_ must be uint64_t");

TEST(ATenPhiloxTest, TypeConsistency) {
  // The static_assert above already fires at compile time; this test
  // confirms the field size at runtime as well.
  EXPECT_EQ(sizeof(at::PhiloxCudaState{}.offset_intragraph_), sizeof(uint64_t));
}

TEST(ATenPhiloxTest, UnpackNonCaptured) {
  constexpr uint64_t kSeed = 42ULL;
  constexpr uint64_t kOffset = 100ULL;
  at::PhiloxCudaState state(kSeed, kOffset);
  EXPECT_FALSE(state.captured_);
  auto [seed, offset] = at::cuda::philox::unpack(state);
  EXPECT_EQ(seed, kSeed);
  EXPECT_EQ(offset, kOffset);
}

TEST(ATenPhiloxTest, DefaultConstructedNotCaptured) {
  at::PhiloxCudaState state;
  EXPECT_FALSE(state.captured_);
  EXPECT_EQ(state.offset_intragraph_, 0ULL);
}

TEST(ATenPhiloxTest, NonCapturedOffsetIntragraphIgnored) {
  // In the non-captured path, offset_intragraph_ plays no role in unpack().
  at::PhiloxCudaState state(7ULL, 13ULL);
  state.offset_intragraph_ = 999ULL;  // should be ignored
  auto [seed, offset] = at::cuda::philox::unpack(state);
  EXPECT_EQ(seed, 7ULL);
  EXPECT_EQ(offset, 13ULL);
}
