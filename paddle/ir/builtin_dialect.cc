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

#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/builtin_op.h"
#include "paddle/ir/builtin_type.h"

namespace ir {
BuiltinDialect::BuiltinDialect(ir::IrContext *context)
    : ir::Dialect(name(), context, ir::TypeId::get<BuiltinDialect>()) {
  initialize();
}

void BuiltinDialect::initialize() {
  // Register all built-in types defined in builtin_type.h.
  RegisterTypes<ir::BFloat16Type,
                ir::Float16Type,
                ir::Float32Type,
                ir::Float64Type,
                ir::Int8Type,
                ir::Int16Type,
                ir::Int32Type,
                ir::Int64Type,
                ir::BoolType,
                ir::VectorType>();

  RegisterAttributes<ir::StrAttribute,
                     ir::BoolAttribute,
                     ir::FloatAttribute,
                     ir::DoubleAttribute,
                     ir::Int32_tAttribute,
                     ir::Int64_tAttribute,
                     ir::ArrayAttribute>();

  RegisterOps<ir::GetParameterOp,
              ir::SetParameterOp,
              ir::CombineOp,
              ir::SliceOp>();
}

}  // namespace ir
