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
#include <map>

#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

TEST(builder_test, type_api) {
  ir::IrContext ctx;
  ir::Builder builder(&ctx);
  EXPECT_EQ(ir::UInt8Type::get(&ctx), builder.uint8_type());
  EXPECT_EQ(ir::Int8Type::get(&ctx), builder.int8_type());
  EXPECT_EQ(ir::VectorType::get(&ctx, std::vector<ir::Type>()),
            builder.vec_type({}));
  EXPECT_EQ(ir::BFloat16Type::get(&ctx), builder.bfloat16_type());
  EXPECT_EQ(ir::Float32Type::get(&ctx), builder.float32_type());
  EXPECT_EQ(ir::Float64Type::get(&ctx), builder.float64_type());
  EXPECT_EQ(ir::IndexType::get(&ctx), builder.index_type());
  EXPECT_EQ(ir::Int16Type::get(&ctx), builder.int16_type());
  EXPECT_EQ(ir::BoolType::get(&ctx), builder.bool_type());
  EXPECT_EQ(ir::Complex64Type::get(&ctx), builder.complex64_type());
  EXPECT_EQ(ir::Complex128Type::get(&ctx), builder.complex128_type());
}

TEST(builder_test, attribute_api) {
  ir::IrContext ctx;
  ir::Builder builder(&ctx);
  EXPECT_EQ(ir::StrAttribute::get(&ctx, "test"), builder.str_attr("test"));
  EXPECT_EQ(ir::BoolAttribute::get(&ctx, true), builder.bool_attr(true));
  EXPECT_EQ(ir::FloatAttribute::get(&ctx, 0.2f), builder.float_attr(0.2f));
  EXPECT_EQ(ir::DoubleAttribute::get(&ctx, 2.0), builder.double_attr(2.0));
  EXPECT_EQ(ir::Int32Attribute::get(&ctx, 2), builder.int32_attr(2));
  EXPECT_EQ(ir::Int64Attribute::get(&ctx, 2), builder.int64_attr(2));
  EXPECT_EQ(ir::ArrayAttribute::get(&ctx, std::vector<ir::Attribute>()),
            builder.array_attr({}));
  EXPECT_EQ(ir::PointerAttribute::get(&ctx, nullptr),
            builder.pointer_attr(nullptr));
}
