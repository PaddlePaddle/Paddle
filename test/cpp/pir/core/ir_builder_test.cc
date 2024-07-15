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

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"

TEST(builder_test, type_api) {
  pir::IrContext ctx;
  pir::Builder builder(&ctx);
  EXPECT_EQ(pir::UInt8Type::get(&ctx), builder.uint8_type());
  EXPECT_EQ(pir::Int8Type::get(&ctx), builder.int8_type());
  EXPECT_EQ(pir::VectorType::get(&ctx, std::vector<pir::Type>()),
            builder.vec_type({}));
  EXPECT_EQ(pir::BFloat16Type::get(&ctx), builder.bfloat16_type());
  EXPECT_EQ(pir::Float32Type::get(&ctx), builder.float32_type());
  EXPECT_EQ(pir::Float64Type::get(&ctx), builder.float64_type());
  EXPECT_EQ(pir::IndexType::get(&ctx), builder.index_type());
  EXPECT_EQ(pir::Int16Type::get(&ctx), builder.int16_type());
  EXPECT_EQ(pir::BoolType::get(&ctx), builder.bool_type());
  EXPECT_EQ(pir::Complex64Type::get(&ctx), builder.complex64_type());
  EXPECT_EQ(pir::Complex128Type::get(&ctx), builder.complex128_type());
  EXPECT_EQ(pir::Float8E4M3FNType::get(&ctx), builder.float8e4m3fn_type());
  EXPECT_EQ(pir::Float8E5M2Type::get(&ctx), builder.float8e5m2_type());
}

TEST(builder_test, attribute_api) {
  pir::IrContext ctx;
  pir::Builder builder(&ctx);
  EXPECT_EQ(pir::StrAttribute::get(&ctx, "test"), builder.str_attr("test"));
  EXPECT_EQ(pir::BoolAttribute::get(&ctx, true), builder.bool_attr(true));
  EXPECT_EQ(pir::FloatAttribute::get(&ctx, 0.2f), builder.float_attr(0.2f));
  EXPECT_EQ(pir::DoubleAttribute::get(&ctx, 2.0), builder.double_attr(2.0));
  EXPECT_EQ(pir::Int32Attribute::get(&ctx, 2), builder.int32_attr(2));
  EXPECT_EQ(pir::Int64Attribute::get(&ctx, 2), builder.int64_attr(2));
  EXPECT_EQ(pir::IndexAttribute::get(&ctx, 2), builder.index_attr(2));
  EXPECT_EQ(pir::ArrayAttribute::get(&ctx, std::vector<pir::Attribute>()),
            builder.array_attr({}));
  EXPECT_EQ(pir::PointerAttribute::get(&ctx, nullptr),
            builder.pointer_attr(nullptr));
}
