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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir_adaptor/translator/type_translator.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/type.h"

template <typename IR_TYPE>
void test_parameterless_type() {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();

  ir::Type type = IR_TYPE::get(ctx);
  VLOG(3) << "[type printer test] type: " << type;
  phi::DataType phi_type = paddle::dialect::TransToPhiDataType(type);
  EXPECT_EQ(type, paddle::dialect::TransToIrDataType(phi_type));

  auto& type_translator = paddle::translator::TypeTranslator::instance();
  paddle::framework::VarDesc empty_var_desc("empty");
  auto proto_type = paddle::framework::TransToProtoVarType(phi_type);
  ir::Type final_type = type_translator[proto_type](ctx, empty_var_desc);
  EXPECT_EQ(type, final_type);
}

template <typename... IR_TYPE>
void test_parameterless_type_helper() {
  (void)std::initializer_list<int>{0,
                                   (test_parameterless_type<IR_TYPE>(), 0)...};
}

TEST(TypeConverterTest, paramterless_type) {
  test_parameterless_type_helper<ir::UInt8Type,
                                 ir::Int8Type,
                                 ir::BFloat16Type,
                                 ir::Float16Type,
                                 ir::Float32Type,
                                 ir::Float64Type,
                                 ir::Int16Type,
                                 ir::Int32Type,
                                 ir::Int64Type,
                                 ir::BoolType,
                                 ir::Complex64Type,
                                 ir::Complex128Type>();
}
