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

#include "paddle/ir/pdll/pdl_dialect/pdl_dialect.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_printer.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_ops.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_types.h"

namespace ir {
namespace pdl {

PDLDialect::PDLDialect(ir::IrContext* context)
    : ir::Dialect(name(), context, ir::TypeId::get<PDLDialect>()) {
  initialize();
}

void PDLDialect::PrintType(ir::Type type, std::ostream& os) const {
  if (TypeType type_type = type.dyn_cast<TypeType>()) {
    os << "!pdl.type";
  } else if (ValueType value_type = type.dyn_cast<ValueType>()) {
    os << "!pdl.value";
  } else if (AttributeType attribute_type = type.dyn_cast<AttributeType>()) {
    os << "!pdl.attribute";
  } else if (OperationType operation_type = type.dyn_cast<OperationType>()) {
    os << "!pdl.operation";
  } else if (RangeType range_type = type.dyn_cast<RangeType>()) {
    auto ele_type = range_type.getElementType();
    os << "!pdl.range<";
    PrintType(ele_type, os);
    os << ">";
  }
}

void PDLDialect::PrintAttribute(ir::Attribute type, std::ostream& os) const {}

void PDLDialect::initialize() {
  RegisterTypes<TypeType, ValueType, AttributeType, OperationType, RangeType>();

  RegisterOps<PDL_PatternOp,
              PDL_OperationOp,
              PDL_OperandOp,
              PDL_TypeOp,
              PDL_AttributeOp,
              PDL_EraseOp,
              PDL_ResultOp,
              PDL_ReplaceOp,
              PDL_ApplyNativeConstraintOp,
              PDL_ApplyNativeRewriteOp,
              PDL_RewriteOp>();

  // RegisterInterfaces<>();
}

void PDLDialect::PrintOperation(Operation* op, IrPrinter& printer) const {
  // if (op->name() == "pdl.type") {
  //   auto& os = printer.os;
  //   auto type_op = op->dyn_cast<pdl::PDL_TypeOp>();
  //   // os << ""
  //   printer.PrintOpResult(op);
  //   os << " =";
  //   os << " \"" << op->name() << "\"";
  //   if (auto attr = type_op->attributes().at("type")) {
  //     os << " : ";
  //     // attr.isa<TypeAttribute>();
  //     os << attr.dyn_cast<TypeAttribute>().GetValue();
  //   }
  //   return;
  // }

  if (op->name() == "pdl.pattern" || op->name() == "pdl.rewrite") {
    printer.PrintGeneralOperation(op);
    if (op->num_regions() > 0) {
      printer.os << "\n";
    }
    for (size_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      printer.PrintRegion(region);
    }
  } else {
    printer.PrintGeneralOperation(op);
  }
}

}  // namespace pdl
}  // namespace ir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDLDialect)
