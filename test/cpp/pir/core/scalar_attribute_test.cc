// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"

using ScalarAttribute = paddle::dialect::ScalarAttribute;

TEST(ScalarTest, base) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  pir::Attribute bool_scalar = pir::BoolAttribute::get(ctx, false);
  EXPECT_TRUE(bool_scalar.isa<ScalarAttribute>());
  EXPECT_TRUE(bool_scalar.isa<pir::BoolAttribute>());
  pir::BoolAttribute pure_bool = bool_scalar.dyn_cast<pir::BoolAttribute>();
  EXPECT_TRUE(pure_bool.isa<ScalarAttribute>());
  ScalarAttribute scalar_from_bool = bool_scalar.dyn_cast<ScalarAttribute>();
  EXPECT_TRUE(scalar_from_bool.isa<pir::BoolAttribute>());
  EXPECT_NO_THROW(scalar_from_bool.dyn_cast<pir::BoolAttribute>());
}

TEST(ScalarTest, test_classof) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Attribute bool_scalar = pir::BoolAttribute::get(ctx, false);
  EXPECT_TRUE(bool_scalar.isa<ScalarAttribute>());

  pir::Attribute float_scalar = pir::FloatAttribute::get(ctx, 1.0f);
  EXPECT_TRUE(float_scalar.isa<ScalarAttribute>());

  pir::Attribute double_scalar = pir::DoubleAttribute::get(ctx, 1.0);
  EXPECT_TRUE(double_scalar.isa<ScalarAttribute>());

  pir::Attribute int32_scalar = pir::Int32Attribute::get(ctx, 1);
  EXPECT_TRUE(int32_scalar.isa<ScalarAttribute>());

  pir::Attribute index_scalar = pir::IndexAttribute::get(ctx, 1l);
  EXPECT_TRUE(index_scalar.isa<ScalarAttribute>());

  pir::Attribute int64_scalar = pir::Int64Attribute::get(ctx, 1l);
  EXPECT_TRUE(int64_scalar.isa<ScalarAttribute>());
}
