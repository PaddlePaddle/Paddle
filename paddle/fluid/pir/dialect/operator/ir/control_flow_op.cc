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

#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

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

void IfOp::Build(pir::Builder &builder,             // NOLINT
                 pir::OperationArgument &argument,  // NOLINT
                 pir::Value cond,
                 std::unique_ptr<pir::Block> &&true_block,
                 std::unique_ptr<pir::Block> &&false_block) {
  VLOG(4) << "Start build IfOp";
  if (true_block && !true_block->empty() &&
      true_block->back()->isa<pir::YieldOp>()) {
    auto *op = true_block->back();
    for (size_t i = 0; i < op->num_operands(); ++i) {
      argument.AddOutput(op->operand(i).type());
    }
  }
  if (false_block && !false_block->empty() &&
      false_block->back()->isa<pir::YieldOp>()) {
    auto *op = false_block->back();
    PADDLE_ENFORCE_EQ(op->num_operands(),
                      argument.output_types.size(),
                      phi::errors::PreconditionNotMet(
                          "The output size of true block and false block must "
                          "be equal. but they are %u and %u, respectively",
                          argument.output_types.size(),
                          op->num_operands()));
    for (size_t i = 0; i < op->num_operands(); ++i) {
      PADDLE_ENFORCE_EQ(
          op->operand(i).type(),
          argument.output_types[i],
          phi::errors::PreconditionNotMet("The output[%d] type of true block "
                                          "and false block must be equal.",
                                          i));
    }
  } else {
    PADDLE_ENFORCE(argument.output_types.empty(),
                   phi::errors::PreconditionNotMet(
                       "The output size of true block and false block must be "
                       "equal. but they are %u and 0, respectively",
                       argument.output_types.size()));
  }
  argument.AddRegion()->push_back(true_block.release());
  argument.AddRegion()->push_back(false_block.release());
  argument.AddInput(cond);
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
                    const std::vector<pir::Value> &inputs) {
  argument.AddInputs(inputs);
  for (auto val : inputs) {
    argument.AddOutput(val.type());
  }
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
