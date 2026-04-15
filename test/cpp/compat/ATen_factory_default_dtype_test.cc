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
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorOptions.h>

#include "gtest/gtest.h"

namespace {

class DefaultDtypeGuard {
 public:
  explicit DefaultDtypeGuard(c10::ScalarType dtype)
      : previous_(c10::get_default_dtype()) {
    c10::set_default_dtype(c10::scalarTypeToTypeMeta(dtype));
  }

  ~DefaultDtypeGuard() { c10::set_default_dtype(previous_); }

 private:
  caffe2::TypeMeta previous_;
};

}  // namespace

TEST(ATenFactoryDefaultDtypeTest, EmptyNulloptDtypeUsesCurrentDefault) {
  DefaultDtypeGuard guard(at::kDouble);

  at::Tensor tensor = at::empty(
      {2, 3}, std::nullopt, at::kStrided, at::kCPU, false, std::nullopt);

  ASSERT_EQ(tensor.scalar_type(), at::kDouble);
  ASSERT_EQ(tensor.sizes(), c10::IntArrayRef({2, 3}));
}

TEST(ATenFactoryDefaultDtypeTest, ArangeOmittedDtypeUsesLongForIntegralInputs) {
  DefaultDtypeGuard guard(at::kDouble);

  at::Tensor end_only_default = at::arange(5);
  at::Tensor start_end_default = at::arange(1, 6);
  at::Tensor start_end_step_default = at::arange(1, 7, 2);
  at::Tensor end_only_nullopt =
      at::arange(5, std::nullopt, std::nullopt, at::kCPU, false);
  at::Tensor start_end_nullopt =
      at::arange(1, 6, std::nullopt, std::nullopt, at::kCPU, false);
  at::Tensor start_end_step_nullopt =
      at::arange(1, 7, 2, std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(end_only_default.scalar_type(), at::kLong);
  ASSERT_EQ(start_end_default.scalar_type(), at::kLong);
  ASSERT_EQ(start_end_step_default.scalar_type(), at::kLong);
  ASSERT_EQ(end_only_nullopt.scalar_type(), at::kLong);
  ASSERT_EQ(start_end_nullopt.scalar_type(), at::kLong);
  ASSERT_EQ(start_end_step_nullopt.scalar_type(), at::kLong);
  ASSERT_EQ(end_only_default.data_ptr<int64_t>()[4], 4);
  ASSERT_EQ(start_end_default.data_ptr<int64_t>()[0], 1);
  ASSERT_EQ(start_end_step_default.data_ptr<int64_t>()[2], 5);
  ASSERT_EQ(end_only_nullopt.data_ptr<int64_t>()[4], 4);
  ASSERT_EQ(start_end_nullopt.data_ptr<int64_t>()[0], 1);
  ASSERT_EQ(start_end_step_nullopt.data_ptr<int64_t>()[2], 5);
}

TEST(ATenFactoryDefaultDtypeTest,
     ArangeOmittedDtypeKeepsLargeInt64InputsExact) {
  constexpr int64_t kStart = (1LL << 53) + 1;
  constexpr int64_t kEnd = kStart + 4;

  at::Tensor by_default = at::arange(kStart, kEnd);
  at::Tensor by_nullopt =
      at::arange(kStart, kEnd, std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(by_default.scalar_type(), at::kLong);
  ASSERT_EQ(by_nullopt.scalar_type(), at::kLong);
  ASSERT_EQ(by_default.numel(), 4);
  ASSERT_EQ(by_nullopt.numel(), 4);

  for (int64_t i = 0; i < 4; ++i) {
    ASSERT_EQ(by_default.data_ptr<int64_t>()[i], kStart + i);
    ASSERT_EQ(by_nullopt.data_ptr<int64_t>()[i], kStart + i);
  }
}

TEST(ATenFactoryDefaultDtypeTest,
     ArangeOmittedDtypeUsesCurrentDefaultForFloatingInputs) {
  DefaultDtypeGuard guard(at::kDouble);

  at::Tensor end_only_default = at::arange(5.0);
  at::Tensor start_end_default = at::arange(1.0, 6.0);
  at::Tensor start_end_step_default = at::arange(1.0, 7.0, 2.0);
  at::Tensor end_only_nullopt =
      at::arange(5.0, std::nullopt, std::nullopt, at::kCPU, false);
  at::Tensor start_end_nullopt =
      at::arange(1.0, 6.0, std::nullopt, std::nullopt, at::kCPU, false);
  at::Tensor start_end_step_nullopt =
      at::arange(1.0, 7.0, 2.0, std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(end_only_default.scalar_type(), at::kDouble);
  ASSERT_EQ(start_end_default.scalar_type(), at::kDouble);
  ASSERT_EQ(start_end_step_default.scalar_type(), at::kDouble);
  ASSERT_EQ(end_only_nullopt.scalar_type(), at::kDouble);
  ASSERT_EQ(start_end_nullopt.scalar_type(), at::kDouble);
  ASSERT_EQ(start_end_step_nullopt.scalar_type(), at::kDouble);
  ASSERT_DOUBLE_EQ(end_only_default.data_ptr<double>()[4], 4.0);
  ASSERT_DOUBLE_EQ(start_end_default.data_ptr<double>()[0], 1.0);
  ASSERT_DOUBLE_EQ(start_end_step_default.data_ptr<double>()[2], 5.0);
  ASSERT_DOUBLE_EQ(end_only_nullopt.data_ptr<double>()[4], 4.0);
  ASSERT_DOUBLE_EQ(start_end_nullopt.data_ptr<double>()[0], 1.0);
  ASSERT_DOUBLE_EQ(start_end_step_nullopt.data_ptr<double>()[2], 5.0);
}

TEST(ATenFactoryDefaultDtypeTest, FullNulloptDtypeUsesCurrentDefault) {
  DefaultDtypeGuard guard(at::kDouble);

  at::Tensor tensor =
      at::full({2, 3}, 1.25, std::nullopt, std::nullopt, at::kCPU, false);
  at::Tensor symint_tensor = at::full_symint(c10::SymIntArrayRef({2, 3}),
                                             2.5,
                                             std::nullopt,
                                             std::nullopt,
                                             at::kCPU,
                                             false);

  ASSERT_EQ(tensor.scalar_type(), at::kDouble);
  ASSERT_EQ(symint_tensor.scalar_type(), at::kDouble);
  ASSERT_DOUBLE_EQ(tensor.data_ptr<double>()[0], 1.25);
  ASSERT_DOUBLE_EQ(symint_tensor.data_ptr<double>()[0], 2.5);
}

TEST(ATenFactoryDefaultDtypeTest, OnesNulloptDtypeUsesCurrentDefault) {
  DefaultDtypeGuard guard(at::kDouble);

  at::Tensor tensor =
      at::ones({2, 3}, std::nullopt, std::nullopt, at::kCPU, false);
  at::Tensor symint_tensor = at::ones_symint(
      c10::SymIntArrayRef({2, 3}), std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(tensor.scalar_type(), at::kDouble);
  ASSERT_EQ(symint_tensor.scalar_type(), at::kDouble);
  ASSERT_DOUBLE_EQ(tensor.data_ptr<double>()[0], 1.0);
  ASSERT_DOUBLE_EQ(symint_tensor.data_ptr<double>()[0], 1.0);
}

TEST(ATenFactoryDefaultDtypeTest, ZerosNulloptDtypeUsesCurrentDefault) {
  DefaultDtypeGuard guard(at::kDouble);

  at::Tensor tensor =
      at::zeros({2, 3}, std::nullopt, at::kStrided, at::kCPU, false);
  at::Tensor symint_tensor = at::zeros_symint(
      c10::SymIntArrayRef({2, 3}), std::nullopt, at::kStrided, at::kCPU, false);

  ASSERT_EQ(tensor.scalar_type(), at::kDouble);
  ASSERT_EQ(symint_tensor.scalar_type(), at::kDouble);
  ASSERT_DOUBLE_EQ(tensor.data_ptr<double>()[0], 0.0);
  ASSERT_DOUBLE_EQ(symint_tensor.data_ptr<double>()[0], 0.0);
}
