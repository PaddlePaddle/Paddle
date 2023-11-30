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
paddle::dialect::IfOp, paddle::dialect::WhileOp, paddle::dialect::HasElementsOp
#else
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/dialect/control_flow/ir/cf_type.h"

using pir::TuplePopOp;
using pir::TuplePushOp;
namespace paddle {
namespace dialect {

void IfOp::Build(pir::Builder &builder,             // NOLINT
                 pir::OperationArgument &argument,  // NOLINT
                 pir::Value cond,
                 std::vector<pir::Type> &&output_types) {
  VLOG(4) << "Start build IfOp";
  argument.AddInput(cond);
  argument.output_types.swap(output_types);
  argument.AddRegion().emplace_back();
  argument.AddRegion().emplace_back();
}

void IfOp::Build(pir::Builder &builder,             // NOLINT
                 pir::OperationArgument &argument,  // NOLINT
                 pir::Value cond,
                 std::unique_ptr<pir::Block> &&true_block,
                 std::unique_ptr<pir::Block> &&false_block) {
  VLOG(4) << "Start build IfOp";
  if (true_block && !true_block->empty() &&
      true_block->back().isa<pir::YieldOp>()) {
    auto &op = true_block->back();
    for (size_t i = 0; i < op.num_operands(); ++i) {
      argument.AddOutput(op.operand(i).type());
    }
  }
  if (false_block && !false_block->empty() &&
      false_block->back().isa<pir::YieldOp>()) {
    auto &op = false_block->back();
    auto size = op.num_operands();
    PADDLE_ENFORCE_EQ(size,
                      argument.output_types.size(),
                      phi::errors::PreconditionNotMet(
                          "The output size of true block and false block must "
                          "be equal. but they are %u and %u, respectively",
                          argument.output_types.size(),
                          size));
    for (size_t i = 0; i < size; ++i) {
      PADDLE_ENFORCE_EQ(
          op.operand(i).type(),
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
  argument.AddRegion().push_back(true_block.release());
  argument.AddRegion().push_back(false_block.release());
  argument.AddInput(cond);
}

pir::Block *IfOp::true_block() {
  pir::Region &region = true_region();
  if (region.empty()) region.emplace_back();
  return &region.front();
}
pir::Block *IfOp::false_block() {
  pir::Region &region = false_region();
  if (region.empty()) region.emplace_back();
  return &region.front();
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
  for (auto &item : *true_block()) {
    os << "\n  ";
    printer.PrintOperation(&item);
  }
  os << "\n } else {";
  for (auto &item : *false_block()) {
    os << "\n  ";
    printer.PrintOperation(&item);
  }
  os << "\n }";
}
void IfOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: IfOp.";
  auto input_size = num_operands();
  PADDLE_ENFORCE_EQ(
      input_size,
      1u,
      phi::errors::PreconditionNotMet(
          "The size %d of inputs must be equal to 1.", input_size));

  if ((*this)->operand_source(0).type().isa<pir::DenseTensorType>()) {
    PADDLE_ENFORCE(
        (*this)
            ->operand_source(0)
            .type()
            .dyn_cast<pir::DenseTensorType>()
            .dtype()
            .isa<pir::BoolType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 1th input, it should be a "
            "bool DenseTensorType."));
  }

  PADDLE_ENFORCE_EQ((*this)->num_regions(),
                    2u,
                    phi::errors::PreconditionNotMet(
                        "The size %d of regions must be equal to 2.",
                        (*this)->num_regions()));
}

void IfOp::VerifyRegion() {
  VLOG(4) << "Start Verifying sub regions for: IfOp.";
  PADDLE_ENFORCE_EQ(
      (*this)->region(0).size(),
      1u,
      phi::errors::PreconditionNotMet("The size %d of true_region must be 1.",
                                      (*this)->region(0).size()));

  if ((*this)->num_results() != 0) {
    PADDLE_ENFORCE_EQ(
        (*this)->region(0).size(),
        (*this)->region(1).size(),
        phi::errors::PreconditionNotMet("The size %d of true_region must be "
                                        "equal to the size %d of false_region.",
                                        (*this)->region(0).size(),
                                        (*this)->region(1).size()));

    auto &true_last_op = (*this)->region(0).front().back();
    auto &false_last_op = (*this)->region(1).front().back();
    PADDLE_ENFORCE_EQ(true,
                      true_last_op.isa<pir::YieldOp>(),
                      phi::errors::PreconditionNotMet(
                          "The last of true block must be YieldOp"));
    PADDLE_ENFORCE_EQ(true_last_op.num_operands(),
                      (*this)->num_results(),
                      phi::errors::PreconditionNotMet(
                          "The size of last of true block op's input must be "
                          "equal to IfOp's outputs num."));
    PADDLE_ENFORCE_EQ(true,
                      false_last_op.isa<pir::YieldOp>(),
                      phi::errors::PreconditionNotMet(
                          "The last of false block must be YieldOp"));
    PADDLE_ENFORCE_EQ(false_last_op.num_operands(),
                      (*this)->num_results(),
                      phi::errors::PreconditionNotMet(
                          "The size of last of false block op's input must be "
                          "equal to IfOp's outputs num."));
  }
}

std::vector<std::vector<pir::OpResult>> IfOp::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs_,
    const std::vector<std::vector<pir::OpResult>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  PADDLE_ENFORCE_EQ(inputs_.size() == 1u && inputs_[0].size() >= 1u,
                    true,
                    phi::errors::InvalidArgument(
                        "if op's inputs' size should be 1, and the inputs[0] "
                        "should be non-empty. "
                        "Now the inputs's size is %d or inputs[0] is empty.",
                        inputs_.size()));

  VLOG(6) << "Prepare inputs for if_grad";
  auto cond_val = inputs_[0][0];
  VLOG(6) << "Prepare attributes for if_grad";

  VLOG(6) << "Prepare outputs for if_grad";

  std::vector<pir::Type> output_types;
  for (size_t i = 0; i < inputs_[0].size(); ++i) {
    if (!stop_gradients[0][i]) {
      output_types.push_back(inputs_[0][i].type());
    }
  }

  auto if_grad = ApiBuilder::Instance().GetBuilder()->Build<IfOp>(
      cond_val, std::move(output_types));

  std::vector<std::vector<pir::OpResult>> res{
      std::vector<pir::OpResult>(inputs_[0].size())};

  for (size_t i = 0, j = 0; i < inputs_[0].size(); ++i) {
    if (!stop_gradients[0][i]) {
      res[0][i] = if_grad->result(j++);
    }
  }
  return res;
}

void WhileOp::Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond,
                    const std::vector<pir::Value> &inputs) {
  argument.AddInput(cond);
  argument.AddInputs(inputs);
  auto &body = argument.AddRegion().emplace_back();
  for (auto val : inputs) {
    argument.AddOutput(val.type());
    body.AddArgument(val.type());
  }
}
pir::Block &WhileOp::body() {
  pir::Region &body_region = (*this)->region(0);
  if (body_region.empty()) body_region.emplace_back();
  return body_region.front();
}
pir::Value WhileOp::cond() { return (*this)->operand_source(0); }

void WhileOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = \"" << name() << "\"(";
  printer.PrintValue(cond());
  os << ") [";
  auto operands = (*this)->operands_source();
  pir::PrintInterleave(
      operands.begin() + 1,
      operands.end(),
      [&](pir::Value v) { printer.PrintValue(v); },
      [&]() { os << ", "; });
  os << "] { \n ^";
  pir::PrintInterleave(
      body().args_begin(),
      body().args_end(),
      [&](pir::Value v) { printer.PrintValue(v); },
      [&]() { os << ", "; });
  for (auto &item : body()) {
    os << "\n  ";
    printer.PrintOperation(&item);
  }
  os << "\n }";
}

std::vector<std::vector<pir::OpResult>> TuplePushOpVjpInterfaceModel::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs,
    const std::vector<std::vector<pir::OpResult>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  PADDLE_ENFORCE_EQ(inputs.size() == 1u && inputs[0].size() >= 1u,
                    true,
                    phi::errors::InvalidArgument(
                        "tupe_push op's inputs' size should be 1, and the "
                        "inputs[0] should be non-empty. "
                        "Now the inputs's size is %d or inputs[0] is empty.",
                        inputs.size()));
  auto pop_op = ApiBuilder::Instance().GetBuilder()->Build<TuplePopOp>(
      TuplePushOp::dyn_cast(op).outlet());
  std::vector<std::vector<pir::OpResult>> res{
      std::vector<pir::OpResult>{nullptr}};
  for (size_t i = 0u; i < pop_op.num_results(); ++i) {
    res[0].push_back(pop_op.result(i));
  }
  return res;
}

void HasElementsOp::Build(pir::Builder &builder,             // NOLINT
                          pir::OperationArgument &argument,  // NOLINT
                          pir::Value stack) {
  argument.AddInput(stack);
  argument.AddOutput(
      DenseTensorType::get(builder.ir_context(), builder.bool_type(), {1}));
}
void HasElementsOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs ,attributes for: HasElementsOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 1u, "The size of inputs must equal to 1.");
  IR_ENFORCE(operand_source(0).type().isa<pir::StackType>(),
             "The first input of cf.has_elements must be stack_type.");

  // No attributes should be verify.

  // Verify outputs:
  IR_ENFORCE(num_results() == 1u, "The size of outputs must be equal to 1.");
  IR_ENFORCE((*this)->result_type(0).isa<DenseTensorType>() ||
                 (*this)->result_type(0).isa<AllocatedDenseTensorType>(),
             "The type of cf.has_elements' output is not correct.");
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::WhileOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::HasElementsOp)

#endif
