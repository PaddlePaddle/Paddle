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

#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/op_base.h"

namespace pir::shape {

const char SymbolAttribute::attr_name[] = "sym_shape_str";  // NOLINT

symbol::ShapeOrDataDimExprs SymbolAttribute::data() const {
  return storage()->data();
}

SymbolAttribute SymbolAttribute::get(pir::IrContext* ctx,
                                     const symbol::ShapeOrDataDimExprs& value) {
  return AttributeManager::get<SymbolAttribute>(ctx, value);
}

void SetShapeAttrForOp(pir::Operation* op,
                       const symbol::ShapeOrDataDimExprs& shape_data) {
  std::ostringstream attr_str;
  attr_str << shape_data;
  op->set_attribute(
      SymbolAttribute::attr_name,
      pir::StrAttribute::get(pir::IrContext::Instance(), attr_str.str()));
}

}  // namespace pir::shape

IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::SymbolAttribute)
