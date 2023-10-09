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
#ifdef GET_OP_LIST
#undef GET_OP_LIST
paddle::dialect::IfOp, paddle::dialect::WhileOp
#else
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"

#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/operation_utils.h"

namespace paddle {
namespace dialect {

void IfOp::Build(pir::Builder &builder,             // NOLINT
                 pir::OperationArgument &argument,  // NOLINT
                 pir::Value cond,
                 std::vector<pir::Type> &&output_types) {
  VLOG(4) << "Start build IfOp";
  argument.AddRegions(2u);
  argument.AddInput(cond);
  argument.output_types.swap(output_types);
}
pir::Block *IfOp::true_block() {
  pir::Region &true_region = (*this)->region(0);
  if (true_region.empty()) true_region.emplace_back();
  return true_region.front();
}
pir::Block *IfOp::false_block() {
  pir::Region &false_region = (*this)->region(1);
  if (false_region.empty()) false_region.emplace_back();
  return false_region.front();
}
void IfOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = pd_op.if";
  printer.PrintOpOperands(op);
  os << " -> ";
  printer.PrintOpReturnType(op);
  os << "{";
  for (auto item : *true_block()) {
    os << "\n  ";
    printer.PrintOperation(item);
  }
  os << "\n } else {";
  for (auto item : *false_block()) {
    os << "\n  ";
    printer.PrintOperation(item);
  }
  os << "\n }";
}
void IfOp::Verify() {}

void WhileOp::Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    const std::vector<pir::Type> &output_types) {
  argument.AddInputs(inputs);
  argument.AddOutputs(output_types);
  argument.AddRegions(2u);
}
pir::Block *WhileOp::cond_block() {
  pir::Region &cond_region = (*this)->region(0);
  if (cond_region.empty()) cond_region.emplace_back();
  return cond_region.front();
}
pir::Block *WhileOp::body_block() {
  pir::Region &body_region = (*this)->region(1);
  if (body_region.empty()) body_region.emplace_back();
  return body_region.front();
}

void WhileOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " \"" << name() << "\"";
  printer.PrintOpOperands(op);
  os << " -> ";
  printer.PrintOpReturnType(op);
  os << "{";
  for (auto item : *cond_block()) {
    os << "\n  ";
    printer.PrintOperation(item);
  }
  os << "\n } do {";
  for (auto item : *body_block()) {
    os << "\n  ";
    printer.PrintOperation(item);
  }
  os << "\n }";
}
}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::WhileOp)

#endif
