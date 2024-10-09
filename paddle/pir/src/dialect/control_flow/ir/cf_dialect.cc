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
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace pir {
void ControlFlowDialect::initialize() {
  RegisterTypes<StackType, InletType, OutletType>();
  RegisterOps<YieldOp, StackCreateOp, TuplePushOp, TuplePopOp>();
}

void ControlFlowDialect::PrintType(pir::Type type, std::ostream& os) const {
  os << name();
  os << '.';
  if (type.isa<StackType>()) {
    os << "stack";
  } else if (type.isa<InletType>()) {
    os << "inlet";
  } else if (type.isa<OutletType>()) {
    os << "outlet";
  } else {
    os << "unknown type";
  }
}

pir::OpPrintFn ControlFlowDialect::PrintOperation(
    const pir::Operation& op) const {
  if (auto create_op = op.dyn_cast<StackCreateOp>()) {
    return [](const pir::Operation& op, pir::IrPrinter& printer) {
      auto create_op = op.dyn_cast<StackCreateOp>();
      create_op.Print(printer);
    };
  }
  return nullptr;
}
}  // namespace pir
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ControlFlowDialect)
