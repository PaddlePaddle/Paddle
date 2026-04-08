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
#include <ATen/ops/equal.h>
#include <ATen/ops/tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "gtest/gtest.h"

TEST(TensorEqualTest, DifferentShapeReturnsFalse) {
  at::Tensor a = at::ones({2, 2}, at::kFloat);
  at::Tensor b = at::ones({2, 3}, at::kFloat);

  ASSERT_FALSE(at::equal(a, b));
  ASSERT_FALSE(a.equal(b));
}

TEST(TensorEqualTest, DtypeMismatchCastsOtherTensor) {
  at::Tensor a = at::tensor({1.0f, 2.0f, 3.0f}, at::kFloat);
  at::Tensor b = at::tensor({1, 2, 3}, at::kInt);

  ASSERT_TRUE(at::equal(a, b));
  ASSERT_TRUE(a.equal(b));
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(TensorEqualTest, DeviceMismatchThrows) {
  at::Tensor cpu = at::ones({2, 2}, at::kFloat);
  at::Tensor gpu =
      at::ones({2, 2}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  ASSERT_THROW((void)at::equal(cpu, gpu), std::exception);
}
#endif
