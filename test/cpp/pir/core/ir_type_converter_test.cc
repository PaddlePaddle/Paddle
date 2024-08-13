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
#include <sstream>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/ir_adaptor/translator/attribute_translator.h"
#include "paddle/fluid/ir_adaptor/translator/type_translator.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/type.h"

template <typename IR_TYPE>
void test_parameterless_type() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Type type = IR_TYPE::get(ctx);
  std::stringstream ss;
  ss << type;
  EXPECT_GT(ss.str().size(), 0u);
  EXPECT_NE(ss.str(), "<<NULL TYPE>>");
  phi::DataType phi_type = paddle::dialect::TransToPhiDataType(type);
  EXPECT_EQ(type, paddle::dialect::TransToIrDataType(phi_type));

  auto& type_translator = paddle::translator::TypeTranslator::instance();
  paddle::framework::VarDesc empty_var_desc("empty");
  auto proto_type = paddle::framework::TransToProtoVarType(phi_type);
  pir::Type final_type = type_translator[proto_type](ctx, empty_var_desc);
  EXPECT_EQ(type, final_type);
}

template <typename... IR_TYPE>
void test_parameterless_type_helper() {
  (void)std::initializer_list<int>{0,
                                   (test_parameterless_type<IR_TYPE>(), 0)...};
}

TEST(TypeConverterTest, parameterless_type) {
  test_parameterless_type_helper<pir::UInt8Type,
                                 pir::Int8Type,
                                 pir::BFloat16Type,
                                 pir::Float16Type,
                                 pir::Float32Type,
                                 pir::Float64Type,
                                 pir::Int16Type,
                                 pir::Int32Type,
                                 pir::Int64Type,
                                 pir::BoolType,
                                 pir::Complex64Type,
                                 pir::Complex128Type,
                                 pir::Float8E4M3FNType,
                                 pir::Float8E5M2Type>();
}

void test_index_type() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Type type = pir::IndexType::get(ctx);
  std::stringstream ss;
  ss << type;
  EXPECT_GT(ss.str().size(), 0u);
  EXPECT_EQ(ss.str(), "index");
  EXPECT_NE(ss.str(), "<<NULL TYPE>>");
  phi::DataType phi_type = paddle::dialect::TransToPhiDataType(type);
  auto& type_translator = paddle::translator::TypeTranslator::instance();
  paddle::framework::VarDesc empty_var_desc("empty");
  auto proto_type = paddle::framework::TransToProtoVarType(phi_type);
  pir::Type final_type = type_translator[proto_type](ctx, empty_var_desc);
  EXPECT_EQ(paddle::dialect::TransToIrDataType(phi_type), final_type);
}

TEST(IndexTypeConverterTest, index_type) { test_index_type(); }

TEST(AttributeConverterTest, int2bool) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  auto& attr_translator = paddle::translator::AttributeTranslator::instance();
  int v = 0;
  paddle::framework::Attribute attr(v);
  pir::Attribute pir_attr = attr_translator("pir::BoolAttribute", attr);
  EXPECT_TRUE(pir_attr.isa<pir::BoolAttribute>());

  int64_t v1 = 0;
  paddle::framework::Attribute attr1(v1);
  pir_attr = attr_translator("pir::BoolAttribute", attr1);
  EXPECT_TRUE(pir_attr.isa<pir::BoolAttribute>());

  pir_attr =
      attr_translator("pir::BoolAttribute", paddle::framework::Attribute());
  EXPECT_TRUE(pir_attr.isa<pir::BoolAttribute>());
}
