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
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ======================== register_hook tests ========================

TEST(TensorHookTest, RegisterHookThrows) {
  // register_hook should throw exception as Paddle doesn't support hooks
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  auto hook = [](const at::Tensor& grad) { return grad; };
  EXPECT_THROW(t.register_hook(hook), std::runtime_error);
}

TEST(TensorHookTest, RegisterHookWithLambda) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Lambda that captures nothing
  EXPECT_THROW(t.register_hook([](const at::Tensor&) { return at::Tensor(); }),
               std::runtime_error);
}

TEST(TensorHookTest, RegisterHookWithMoveOnly) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Move-only lambda
  EXPECT_THROW(
      t.register_hook([](const at::Tensor&) mutable { return at::Tensor(); }),
      std::runtime_error);
}
