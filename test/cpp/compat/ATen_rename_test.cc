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

// ======================== rename tests ========================

TEST(TensorRenameTest, RenameWithNames) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  std::vector<std::string> names = {"height", "width"};
  at::DimnameList name_list(names);
  at::Tensor result = t.rename(name_list);

  ASSERT_EQ(result.sizes(), t.sizes());
}

TEST(TensorRenameTest, RenameNone) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  at::Tensor result = t.rename(::std::nullopt);

  ASSERT_EQ(result.sizes(), t.sizes());
}

TEST(TensorRenameTest, RenamePreservesData) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  std::vector<std::string> names = {"a", "b"};
  at::Tensor result = t.rename(names);

  // Data should be preserved
  ASSERT_EQ(result.numel(), t.numel());
  for (int i = 0; i < t.numel(); i++) {
    ASSERT_FLOAT_EQ(result.data_ptr<float>()[i], t.data_ptr<float>()[i]);
  }
}
