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

#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/type_support.h"

TEST(ir_context, type) {
  // Test creation of built-in singleton type.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Type fp32_1 = ir::Float32Type::get(ctx);

  // Test interfaces of class Type
  ir::Type fp32_2 = ir::Float32Type::get(ctx);
  EXPECT_EQ(fp32_1 == fp32_2, 1);
  EXPECT_EQ(fp32_1 != fp32_2, 0);
  EXPECT_EQ(fp32_1.type_id() == fp32_2.type_id(), 1);
  EXPECT_EQ(&fp32_1.abstract_type() ==
                &ir::AbstractType::lookup(fp32_1.type_id(), ctx),
            1);
  EXPECT_EQ(ir::Float32Type::classof(fp32_1), 1);

  ir::Type int1_1 = ir::IntegerType::get(ctx, 1, 0);
  ir::Type int1_2 = ir::IntegerType::get(ctx, 1, 0);
  EXPECT_EQ(int1_1 == int1_2, 1);
}

TEST(ir_context, type_beta) {
  // Test creation of built-in singleton type.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Type fp32_beta_1 = ir::Float32TypeBeta::get(ctx);

  // Test interfaces of class Type
  ir::Type fp32_beta_2 = ir::Float32TypeBeta::get(ctx);
  EXPECT_EQ(fp32_beta_1 == fp32_beta_2, 1);
  EXPECT_EQ(fp32_beta_1 != fp32_beta_2, 0);
  EXPECT_EQ(fp32_beta_1.type_id() == fp32_beta_2.type_id(), 1);
  EXPECT_EQ(&fp32_beta_1.abstract_type() ==
                &ir::AbstractType::lookup(fp32_beta_1.type_id(), ctx),
            1);
  EXPECT_EQ(ir::Float32TypeBeta::classof(fp32_beta_1), 1);
  EXPECT_EQ(ir::Float32Type::classof(fp32_beta_1), 0);
}
