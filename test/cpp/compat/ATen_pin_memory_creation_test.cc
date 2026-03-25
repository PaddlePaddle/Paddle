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
#include <ATen/ops/arange.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>

#include <optional>

#include "gtest/gtest.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

namespace {

void AssertPinned(const at::Tensor& t) {
  ASSERT_TRUE(t.is_pinned());
  ASSERT_FALSE(t.is_cuda());
}

void AssertNotPinned(const at::Tensor& t) { ASSERT_FALSE(t.is_pinned()); }

}  // namespace

TEST(ATenPinMemoryCreationTest, FullPinMemory) {
  // Test using TensorOptions with pinned_memory
  auto by_options = at::full(
      {2, 3}, 1.5f, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  AssertPinned(by_options);

  // Test using explicit arguments with CPU device (should succeed)
  auto by_args =
      at::full({2, 3}, 1.5f, at::kFloat, std::nullopt, at::kCPU, true);
  AssertPinned(by_args);

  // Test without pin_memory
  auto no_pin =
      at::full({2, 3}, 1.5f, at::kFloat, std::nullopt, at::kCPU, false);
  AssertNotPinned(no_pin);
}

TEST(ATenPinMemoryCreationTest, FullPinMemoryWithCUDADeviceErrors) {
  // Test that pin_memory=true with explicit CUDA device throws error
  ASSERT_THROW(
      at::full({2, 3}, 1.5f, at::kFloat, std::nullopt, at::kCUDA, true),
      std::exception);
}

TEST(ATenPinMemoryCreationTest, OnesPinMemory) {
  auto by_options = at::ones(
      {4, 2}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  AssertPinned(by_options);

  auto by_args = at::ones({4, 2}, at::kFloat, std::nullopt, at::kCPU, true);
  AssertPinned(by_args);

  auto no_pin = at::ones({4, 2}, at::kFloat, std::nullopt, at::kCPU, false);
  AssertNotPinned(no_pin);
}

TEST(ATenPinMemoryCreationTest, OnesPinMemoryWithCUDADeviceErrors) {
  ASSERT_THROW(at::ones({4, 2}, at::kFloat, std::nullopt, at::kCUDA, true),
               std::exception);
}

TEST(ATenPinMemoryCreationTest, ZerosPinMemory) {
  auto by_options = at::zeros(
      {3, 5}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  AssertPinned(by_options);

  auto by_args = at::zeros({3, 5}, at::kFloat, at::kStrided, at::kCPU, true);
  AssertPinned(by_args);

  auto no_pin = at::zeros({3, 5}, at::kFloat, at::kStrided, at::kCPU, false);
  AssertNotPinned(no_pin);
}

TEST(ATenPinMemoryCreationTest, ZerosPinMemoryWithCUDADeviceErrors) {
  ASSERT_THROW(at::zeros({3, 5}, at::kFloat, at::kStrided, at::kCUDA, true),
               std::exception);
}

TEST(ATenPinMemoryCreationTest, EyePinMemory) {
  auto by_options =
      at::eye(6, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  AssertPinned(by_options);

  auto by_args = at::eye(6, at::kFloat, std::nullopt, at::kCPU, true);
  AssertPinned(by_args);

  auto no_pin = at::eye(6, at::kFloat, std::nullopt, at::kCPU, false);
  AssertNotPinned(no_pin);
}

TEST(ATenPinMemoryCreationTest, EyePinMemoryWithCUDADeviceErrors) {
  ASSERT_THROW(at::eye(6, at::kFloat, std::nullopt, at::kCUDA, true),
               std::exception);
}

TEST(ATenPinMemoryCreationTest, ArangePinMemory) {
  auto by_options = at::arange(
      0, 10, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  AssertPinned(by_options);

  auto by_args = at::arange(0, 10, at::kFloat, std::nullopt, at::kCPU, true);
  AssertPinned(by_args);

  auto no_pin = at::arange(0, 10, at::kFloat, std::nullopt, at::kCPU, false);
  AssertNotPinned(no_pin);
}

TEST(ATenPinMemoryCreationTest, ArangePinMemoryWithCUDADeviceErrors) {
  ASSERT_THROW(at::arange(0, 10, at::kFloat, std::nullopt, at::kCUDA, true),
               std::exception);
}

TEST(ATenPinMemoryCreationTest, EmptyLikePinMemory) {
  auto base = at::ones({2, 4}, at::kFloat);

  auto by_options =
      at::empty_like(base,
                     at::TensorOptions().dtype(at::kFloat).pinned_memory(true),
                     std::nullopt);
  AssertPinned(by_options);

  auto by_args = at::empty_like(
      base, at::kFloat, at::kStrided, at::kCPU, true, std::nullopt);
  AssertPinned(by_args);

  auto no_pin = at::empty_like(
      base, at::kFloat, at::kStrided, at::kCPU, false, std::nullopt);
  AssertNotPinned(no_pin);
}

TEST(ATenPinMemoryCreationTest, EmptyLikePinMemoryWithCUDADeviceErrors) {
  auto base = at::ones({2, 4}, at::kFloat);
  ASSERT_THROW(at::empty_like(base,
                              at::TensorOptions()
                                  .dtype(at::kFloat)
                                  .device(at::kCUDA)
                                  .pinned_memory(true),
                              std::nullopt),
               std::exception);
}

TEST(ATenPinMemoryCreationTest, ZerosLikePinMemory) {
  auto base = at::ones({2, 4}, at::kFloat);

  auto by_options =
      at::zeros_like(base,
                     at::TensorOptions().dtype(at::kFloat).pinned_memory(true),
                     std::nullopt);
  AssertPinned(by_options);

  auto by_args = at::zeros_like(
      base, at::kFloat, at::kStrided, at::kCPU, true, std::nullopt);
  AssertPinned(by_args);

  auto no_pin = at::zeros_like(
      base, at::kFloat, at::kStrided, at::kCPU, false, std::nullopt);
  AssertNotPinned(no_pin);
}

TEST(ATenPinMemoryCreationTest, ZerosLikePinMemoryWithCUDADeviceErrors) {
  auto base = at::ones({2, 4}, at::kFloat);
  ASSERT_THROW(at::zeros_like(base,
                              at::TensorOptions()
                                  .dtype(at::kFloat)
                                  .device(at::kCUDA)
                                  .pinned_memory(true),
                              std::nullopt),
               std::exception);
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
