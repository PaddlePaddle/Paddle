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

#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_op.h"

namespace pir::shape {
ShapeDialect::ShapeDialect(IrContext *context)
    : Dialect(name(), context, TypeId::get<ShapeDialect>()) {
  initialize();
}

void ShapeDialect::initialize() {
  RegisterOps<DimOp>();

  RegisterAttributes<SymbolAttribute>();
}

void ShapeDialect::PrintAttribute(pir::Attribute attr, std::ostream &os) const {
  return;
}

}  // namespace pir::shape

IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ShapeDialect)
