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
paddle::dialect::IfOp, paddle::dialect::WhileOp, paddle::dialect::HasElementsOp,
    paddle::dialect::AssertOp, paddle::dialect::SelectInputOp,
    paddle::dialect::SelectOutputOp
#else
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_trait.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/dialect/control_flow/ir/cf_type.h"

using pir::TuplePopOp;
using pir::TuplePushOp;
constexpr char kStopGradientAttrName[] = "stop_gradient";
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
  cond.set_attribute(kStopGradientAttrName, builder.bool_attr(true));
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

    std::vector<pir::Attribute> outs_stop_gradient;
    for (size_t i = 0; i < op.num_operands(); ++i) {
      argument.AddOutput(op.operand(i).type());
      auto bool_attr = op.operand_source(i).attribute<pir::BoolAttribute>(
          kStopGradientAttrName);
      outs_stop_gradient.push_back(bool_attr ? bool_attr
                                             : builder.bool_attr(false));
    }

    argument.AddAttribute(
        kStopGradientAttrName,
        pir::ArrayAttribute::get(builder.ir_context(), outs_stop_gradient));
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
      if (op.operand(i).type() != argument.output_types[i]) {
        auto l_type = op.operand(i).type().dyn_cast<pir::DenseTensorType>();
        auto r_type = argument.output_types[i].dyn_cast<pir::DenseTensorType>();
        PADDLE_ENFORCE_EQ(l_type && r_type,
                          true,
                          phi::errors::PreconditionNotMet(
                              "The output[%d] of true_block&false_block must "
                              "be dense tensor type.",
                              i));
        PADDLE_ENFORCE_EQ(l_type.dtype(),
                          r_type.dtype(),
                          phi::errors::PreconditionNotMet(
                              "The dtype in output[%d] of "
                              "true_block&false_block must be equal.",
                              i));
        if (l_type.data_layout() != phi::DataLayout::UNDEFINED &&
            r_type.data_layout() != phi::DataLayout::UNDEFINED) {
          PADDLE_ENFORCE_EQ(
              l_type.data_layout(),
              r_type.data_layout(),
              phi::errors::PreconditionNotMet(
                  "The data_layout in output[%d] of "
                  "true_block (%s) & false_block (%s) must be equal.",
                  i,
                  l_type.data_layout(),
                  r_type.data_layout()));
        }
        PADDLE_ENFORCE_EQ(l_type.lod(),
                          r_type.lod(),
                          phi::errors::PreconditionNotMet(
                              "The lod in output[%d] of true_block&false_block "
                              "must be equal.",
                              i));
        PADDLE_ENFORCE_EQ(l_type.offset(),
                          r_type.offset(),
                          phi::errors::PreconditionNotMet(
                              "The offset in output[%d] of "
                              "true_block&false_block must be equal.",
                              i));
        auto dim = common::ComputeCompatibleDim(l_type.dims(), r_type.dims());
        auto new_type = DenseTensorType::get(builder.ir_context(),
                                             l_type.dtype(),
                                             dim,
                                             l_type.data_layout(),
                                             l_type.lod(),
                                             l_type.offset());
        argument.output_types[i] = new_type;
      }
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
  cond.set_attribute(kStopGradientAttrName, builder.bool_attr(true));
}

pir::Block &IfOp::true_block() {
  pir::Region &region = true_region();
  if (region.empty()) region.emplace_back();
  return region.front();
}
pir::Block &IfOp::false_block() {
  pir::Region &region = false_region();
  if (region.empty()) region.emplace_back();
  return region.front();
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
  for (auto &item : true_block()) {
    os << "\n  ";
    printer.PrintOperation(&item);
  }
  os << "\n } else {";
  for (auto &item : false_block()) {
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
  VLOG(4) << "Start Verifying true branch.";
  PADDLE_ENFORCE_EQ(
      (*this)->region(0).size(),
      1u,
      phi::errors::PreconditionNotMet("The size %d of true_region must be 1.",
                                      (*this)->region(0).size()));
  if ((*this)->num_results() != 0) {
    auto &true_block = (*this)->region(0).front();
    PADDLE_ENFORCE_GT(
        true_block.size(),
        0u,
        phi::errors::PreconditionNotMet(
            "The true block must have at least one op yield op."));
    auto &true_last_op = true_block.back();
    PADDLE_ENFORCE_EQ(true,
                      true_last_op.isa<pir::YieldOp>(),
                      phi::errors::PreconditionNotMet(
                          "The last of true block must be YieldOp"));
    PADDLE_ENFORCE_EQ(true_last_op.num_operands(),
                      (*this)->num_results(),
                      phi::errors::PreconditionNotMet(
                          "The size of last of true block op's input must be "
                          "equal to IfOp's outputs num."));
    VLOG(4) << "Start Verifying false branch.";
    PADDLE_ENFORCE_EQ((*this)->region(1).size(),
                      1u,
                      phi::errors::PreconditionNotMet(
                          "The size %d of false_region must be 1.",
                          (*this)->region(0).size()));
    auto &false_block = (*this)->region(1).front();
    PADDLE_ENFORCE_GT(
        false_block.size(),
        0u,
        phi::errors::PreconditionNotMet(
            "The false block must have at least one op yield op."));
    auto &false_last_op = false_block.back();
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

std::vector<std::vector<pir::Value>> IfOp::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs_,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs_.size() >= 1u,
      true,
      phi::errors::InvalidArgument("if op's inputs' size should greater_equal "
                                   "to 1, and all the inputs[i] "
                                   "should be 1 size. "
                                   "Now the inputs's size is %d .",
                                   inputs_.size()));

  VLOG(6) << "Prepare inputs for if_grad";
  auto cond_val = inputs_[0][0];
  VLOG(6) << "Prepare attributes for if_grad";

  VLOG(6) << "Prepare outputs for if_grad";

  std::vector<pir::Type> output_types;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (!stop_gradients[i][0]) {
      output_types.push_back(inputs_[i][0].type());
    }
  }

  auto if_grad = ApiBuilder::Instance().GetBuilder()->Build<IfOp>(
      cond_val, std::move(output_types));

  std::vector<std::vector<pir::Value>> res{inputs_.size()};
  for (size_t i = 0, j = 0; i < inputs_.size(); ++i) {
    res[i].resize(1);
    if (!stop_gradients[i][0]) {
      res[i][0] = if_grad->result(j++);
    }
  }
  return res;
}

void WhileOp::Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond,
                    const std::vector<pir::Value> &inputs,
                    bool construct_body) {
  argument.AddInput(cond);
  argument.AddInputs(inputs);
  std::vector<pir::Attribute> outs_stop_gradient;
  if (construct_body) {
    auto &body = argument.AddRegion().emplace_back();
    for (auto val : inputs) {
      argument.AddOutput(val.type());
      auto arg = body.AddArgument(val.type());
      auto bool_attr = val.attribute<pir::BoolAttribute>(kStopGradientAttrName);
      outs_stop_gradient.push_back(bool_attr ? bool_attr
                                             : builder.bool_attr(false));
      arg.set_attribute(kStopGradientAttrName,
                        bool_attr ? bool_attr : builder.bool_attr(false));
    }
  } else {
    argument.AddRegion(nullptr);
    for (auto val : inputs) {
      argument.AddOutput(val.type());
      auto bool_attr = val.attribute<pir::BoolAttribute>(kStopGradientAttrName);
      outs_stop_gradient.push_back(bool_attr ? bool_attr
                                             : builder.bool_attr(false));
    }
  }

  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(builder.ir_context(), outs_stop_gradient));

  cond.set_attribute(kStopGradientAttrName, builder.bool_attr(true));
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
  os << " = \"" << name() << "\"(cond=";
  printer.PrintValue(cond());
  os << ", inputs=";
  auto operands = (*this)->operands_source();
  pir::PrintInterleave(
      operands.begin() + 1,
      operands.end(),
      [&](pir::Value v) { printer.PrintValue(v); },
      [&]() { os << ", "; });
  os << ") { \n ^";
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

void WhileOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: WhileOp.";
  auto input_size = num_operands();
  PADDLE_ENFORCE_GE(
      input_size,
      1u,
      phi::errors::PreconditionNotMet(
          "The size %d of inputs must be greater or equal to 1.", input_size));

  if (auto cond_type = operand_type(0).dyn_cast<pir::DenseTensorType>()) {
    PADDLE_ENFORCE_EQ(
        cond_type.dtype().isa<pir::BoolType>(),
        true,
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th input, it should be a "
            "bool DenseTensorType."));
  } else if (auto cond_type =
                 operand_type(0).dyn_cast<AllocatedDenseTensorType>()) {
    PADDLE_ENFORCE_EQ(
        cond_type.dtype().isa<pir::BoolType>(),
        true,
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th input, it should be a "
            "bool DenseTensorType."));
  } else {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "Currently,  the while op cond input only support bool dense_tensor "
        "and bool allocated_dense_tensor."));
  }
  PADDLE_ENFORCE_EQ((*this)->num_regions(),
                    1u,
                    phi::errors::PreconditionNotMet(
                        "The size %d of regions must be equal to 1.",
                        (*this)->num_regions()));
  auto output_size = num_results();
  PADDLE_ENFORCE_EQ(output_size + 1,
                    input_size,
                    phi::errors::PreconditionNotMet(
                        "The result size (%d) not equal to input size(%d) + 1.",
                        num_results(),
                        input_size));
  for (size_t index = 0; index < output_size; ++index) {
    PADDLE_ENFORCE_EQ(
        operand_type(index + 1),
        result_type(index),
        phi::errors::PreconditionNotMet(
            "The (%d) result and operand type is not equal.", index));
  }
}

void WhileOp::VerifyRegion() {
  VLOG(4) << "Start verifying sub regions for: WhileOp.";
  PADDLE_ENFORCE_EQ(
      (*this)->region(0).size(),
      1u,
      phi::errors::PreconditionNotMet("The size %d of body_region must be 1.",
                                      (*this)->region(0).size()));
  auto &body_block = body();
  auto output_size = num_results();
  PADDLE_ENFORCE_EQ(
      body_block.args_size(),
      output_size,
      phi::errors::PreconditionNotMet(
          "The result size (%d) not equal to block args size(%d) + 1.",
          output_size,
          body_block.args_size()));

  PADDLE_ENFORCE_EQ(
      body_block.empty(),
      false,
      phi::errors::PreconditionNotMet("The body block is empty."));

  auto yield_op = body_block.back().dyn_cast<pir::YieldOp>();
  auto input_size = num_operands();
  PADDLE_ENFORCE_EQ(
      yield_op && yield_op.num_operands() == input_size,
      true,
      phi::errors::PreconditionNotMet(
          "The body block yield size not equal to operands size."));
  // Todo: fix other bugs and make the following code work.
  // for (size_t index = 0; index < input_size; ++index) {
  //   PADDLE_ENFORCE_EQ(
  //       operand_type(index),
  //       yield_op.operand_type(index),
  //       phi::errors::PreconditionNotMet(
  //           "The (%d) operand and block yield type is not equal.", index));
  // }
  VLOG(4) << "Successful end verifying sub regions for: WhileOp.";
}

std::vector<std::vector<pir::Value>> WhileOp::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  auto fwd_op = WhileOp::dyn_cast(op);
  PADDLE_ENFORCE_NE(
      fwd_op,
      nullptr,
      phi::errors::InvalidArgument("The input op used to called WhileOp::vjp "
                                   "must be non-nullptr while_op"));
  TuplePushOp push_op;
  for (auto iter = fwd_op.body().rbegin(); iter != fwd_op.body().rend();
       ++iter) {
    if (iter->isa<TuplePushOp>()) {
      push_op = iter->dyn_cast<TuplePushOp>();
      PADDLE_ENFORCE_EQ(push_op.container().use_empty(),
                        true,
                        phi::errors::InvalidArgument(
                            "The last container in forward while op must used "
                            "empty while construct while_grad op"));
      break;
    }
  }
  PADDLE_ENFORCE_NE(push_op,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The forward WhileOp must include TuplePushOp, denying "
                        "that we can't construct a reverse loop condition."));

  PADDLE_ENFORCE_GT(inputs.size(),
                    outputs.size(),
                    phi::errors::InvalidArgument(
                        "while op's inputs' size should greater than "
                        "outputs' size, Now the inputs's size is %d ."
                        "the outputs size is %d.",
                        inputs.size(),
                        outputs.size()));
  PADDLE_ENFORCE_EQ(inputs.size(),
                    out_grads.size() + 1,
                    phi::errors::InvalidArgument(
                        "while op's inputs' size should equal to "
                        "output_grads' size + 1, Now the inputs's size is %d ."
                        "the output_grads size is %d.",
                        inputs.size(),
                        out_grads.size()));
  PADDLE_ENFORCE_EQ(stop_gradients[0][0],
                    true,
                    phi::errors::InvalidArgument(
                        "The stop_gradient of condition input must be true."));

  auto &builder = *ApiBuilder::Instance().GetBuilder();
  auto cond_val = builder.Build<HasElementsOp>(push_op.container()).out();

  std::vector<pir::Type> output_types;
  std::vector<pir::Value> loop_vars;

  for (size_t index = 0; index < out_grads.size(); ++index) {
    if (!stop_gradients[index + 1][0]) {
      loop_vars.push_back(out_grads[index][0]);
    }
  }
  auto while_grad = builder.Build<WhileOp>(cond_val, loop_vars);

  std::vector<std::vector<pir::Value>> res(inputs.size());
  for (size_t i = 0, j = 0; i < inputs.size(); ++i) {
    res[i].push_back(stop_gradients[i][0] ? nullptr : while_grad.result(j++));
  }
  return res;
}
std::vector<std::vector<pir::Value>> TuplePushOpVjpInterfaceModel::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs.size() >= 1u,
      true,
      phi::errors::InvalidArgument(
          "tupe_push op's inputs' size should be greater_equal than 1, and the "
          "inputs[i] should be non-empty. "
          "Now the inputs's size is %d.",
          inputs.size()));
  auto pop_op = ApiBuilder::Instance().GetBuilder()->Build<TuplePopOp>(
      TuplePushOp::dyn_cast(op).outlet());
  std::vector<std::vector<pir::Value>> res{inputs.size()};
  res[0].resize(1);
  for (size_t i = 1u; i < inputs.size(); ++i) {
    res[i].resize(1);
    res[i][0] = pop_op.result(i - 1);
  }
  return res;
}

void HasElementsOp::Build(pir::Builder &builder,             // NOLINT
                          pir::OperationArgument &argument,  // NOLINT
                          pir::Value container) {
  argument.AddInput(container);
  argument.AddOutput(
      DenseTensorType::get(builder.ir_context(), builder.bool_type(), {1}));
  std::vector<pir::Attribute> outs_stop_gradient{builder.bool_attr(true)};
  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}
void HasElementsOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs ,attributes for: HasElementsOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 1u, "The size of inputs must equal to 1.");
  IR_ENFORCE(operand_type(0).isa<pir::ContainerType>(),
             "The first input of cf.has_elements must be container type.");

  // No attributes should be verify.

  // Verify outputs:
  IR_ENFORCE(num_results() == 1u, "The size of outputs must be equal to 1.");
  IR_ENFORCE((*this)->result_type(0).isa<DenseTensorType>() ||
                 (*this)->result_type(0).isa<AllocatedDenseTensorType>(),
             "The type of cf.has_elements' output is not correct.");
}

const char *AssertOp::attributes_name[1] = {"summarize"};

void AssertOp::Build(pir::Builder &builder,             // NOLINT
                     pir::OperationArgument &argument,  // NOLINT
                     pir::Value cond_,
                     pir::Value data_,
                     int64_t summarize) {
  VLOG(4) << "Start build AssertOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {cond_, data_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_summarize =
      pir::Int64Attribute::get(pir::IrContext::Instance(), summarize);
  argument.AddAttribute("summarize", attr_summarize);
}

OpInfoTuple AssertOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("cond",
                                   "paddle::dialect::DenseTensorType",
                                   false,
                                   false,
                                   false,
                                   false),
      paddle::dialect::OpInputInfo(
          "data",
          "pir::VectorType<paddle::dialect::DenseTensorType>",
          false,
          false,
          false,
          false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("summarize", "pir::Int64Attribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {};
  paddle::dialect::OpRunTimeInfo run_time_info = paddle::dialect::OpRunTimeInfo(
      "", {""}, "assert", {"cond", "data", "summarize"}, {"cond"}, {}, {}, {});
  return std::make_tuple(inputs, attributes, outputs, run_time_info, "assert");
}

void AssertOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: AssertOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 2u,
               "The size %d of inputs must be equal to 2.",
               input_size);

    if ((*this)->operand_source(0).type().isa<pir::DenseTensorType>()) {
      IR_ENFORCE((*this)
                     ->operand_source(0)
                     .type()
                     .dyn_cast<pir::DenseTensorType>()
                     .dtype()
                     .isa<pir::BoolType>(),
                 "Type validation failed for the 0th input, it should be a "
                 "bool DenseTensorType.");
    }

    if (auto vec_type =
            (*this)->operand(1).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        IR_ENFORCE(vec_type[i].isa<paddle::dialect::DenseTensorType>() ||
                       vec_type[i].isa<paddle::dialect::SelectedRowsType>() ||
                       vec_type[i].isa<AllocatedDenseTensorType>(),
                   "Type validation failed for the 1th input.");
      }
    } else {
      IR_ENFORCE(
          (*this)->operand(1).type().isa<paddle::dialect::DenseTensorType>() ||
              (*this)
                  ->operand(1)
                  .type()
                  .isa<paddle::dialect::SelectedRowsType>(),
          (*this)->operand(1).type().isa<AllocatedDenseTensorType>(),
          "Type validation failed for the 1th input.");
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    IR_ENFORCE(attributes.count("summarize") > 0, "summarize does not exist.");
    IR_ENFORCE(attributes.at("summarize").isa<pir::Int64Attribute>(),
               "Type of attribute: summarize is not pir::Int64Attribute.");
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 0u,
               "The size %d of outputs must be equal to 0.",
               output_size);
    // Outputs num is 0, not need to check outputs type.
  }
  VLOG(4) << "End Verifying for: AssertOp.";
}

void SelectInputOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SelectInputOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto in_size = num_operands();
    IR_ENFORCE(in_size == 3u, "Size %d of inputs must be 3.", in_size);
    auto input1 = (*this)->operand_source(1).type();
    auto input2 = (*this)->operand_source(2).type();
    if (input1.isa<paddle::dialect::DenseTensorType>() &&
        input2.isa<paddle::dialect::DenseTensorType>()) {
      auto tensor1 = input1.dyn_cast<paddle::dialect::DenseTensorType>();
      auto tensor2 = input2.dyn_cast<paddle::dialect::DenseTensorType>();
      IR_ENFORCE(
          tensor1.dtype() == tensor2.dtype(),
          "The 1st input dtype %s should be equal to 2ed input dtype %s.",
          tensor1.dtype(),
          tensor2.dtype());
      IR_ENFORCE(tensor1.data_layout() == tensor2.data_layout(),
                 "The 1st input data_layout %s should be equal to 2ed input "
                 "data_layout %s.",
                 tensor1.data_layout(),
                 tensor2.data_layout());
      IR_ENFORCE(tensor1.lod() == tensor2.lod(),
                 "The 1st input lod %s should be equal to 2ed input lod %s.",
                 tensor1.lod(),
                 tensor2.lod());
      IR_ENFORCE(
          tensor1.offset() == tensor2.offset(),
          "The 1st input offset %s should be equal to 2ed input offset %s.",
          tensor1.offset(),
          tensor2.offset());
    } else if (input1.isa<paddle::dialect::AllocatedDenseTensorType>() &&
               input2.isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto tensor1 =
          input1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      auto tensor2 =
          input1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      IR_ENFORCE(
          tensor1.dtype() == tensor2.dtype(),
          "The 1st input dtype %s should be equal to 2ed input dtype %s.",
          tensor1.dtype(),
          tensor2.dtype());
      IR_ENFORCE(tensor1.data_layout() == tensor2.data_layout(),
                 "The 1st input data_layout %s should be equal to 2ed input "
                 "data_layout %s.",
                 tensor1.data_layout(),
                 tensor2.data_layout());
      IR_ENFORCE(tensor1.lod() == tensor2.lod(),
                 "The 1st input lod %s should be equal to 2ed input lod %s.",
                 tensor1.lod(),
                 tensor2.lod());
      IR_ENFORCE(
          tensor1.offset() == tensor2.offset(),
          "The 1st input offset %s should be equal to 2ed input offset %s.",
          tensor1.offset(),
          tensor2.offset());
      IR_ENFORCE(
          tensor1.place() == tensor2.place(),
          "The 1st input place %s should be equal to 2ed input place %s.",
          tensor1.place(),
          tensor2.place());
    } else {
      IR_ENFORCE(input1 == input2,
                 "The 1st input type %s should be equal to 2ed input type %s.",
                 input1,
                 input2);
    }
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto out_size = num_results();
    IR_ENFORCE(
        out_size == 1u, "Size %d of outputs must be equal to 1.", out_size);
  }
  VLOG(4) << "End Verifying for: AssignArray_Op.";
}

void SelectOutputOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SelectOutputOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto in_size = num_operands();
    IR_ENFORCE(in_size == 2u, "Size %d of inputs must be 2.", in_size);
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto out_size = num_results();
    IR_ENFORCE(
        out_size == 2u, "Size %d of outputs must be equal to 2.", out_size);

    auto out1 = (*this)->result(0).type();
    auto out2 = (*this)->result(1).type();
    if (out1.isa<paddle::dialect::DenseTensorType>() &&
        out2.isa<paddle::dialect::DenseTensorType>()) {
      auto tensor1 = out1.dyn_cast<paddle::dialect::DenseTensorType>();
      auto tensor2 = out2.dyn_cast<paddle::dialect::DenseTensorType>();
      IR_ENFORCE(
          tensor1.dtype() == tensor2.dtype(),
          "The 1st input dtype %s should be equal to 2ed input dtype %s.",
          tensor1.dtype(),
          tensor2.dtype());
      IR_ENFORCE(tensor1.data_layout() == tensor2.data_layout(),
                 "The 1st input data_layout %s should be equal to 2ed input "
                 "data_layout %s.",
                 tensor1.data_layout(),
                 tensor2.data_layout());
      IR_ENFORCE(tensor1.lod() == tensor2.lod(),
                 "The 1st input lod %s should be equal to 2ed input lod %s.",
                 tensor1.lod(),
                 tensor2.lod());
      IR_ENFORCE(
          tensor1.offset() == tensor2.offset(),
          "The 1st input offset %s should be equal to 2ed input offset %s.",
          tensor1.offset(),
          tensor2.offset());
    } else if (out1.isa<paddle::dialect::AllocatedDenseTensorType>() &&
               out2.isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto tensor1 = out1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      auto tensor2 = out2.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      IR_ENFORCE(
          tensor1.dtype() == tensor2.dtype(),
          "The 1st input dtype %s should be equal to 2ed input dtype %s.",
          tensor1.dtype(),
          tensor2.dtype());
      IR_ENFORCE(tensor1.data_layout() == tensor2.data_layout(),
                 "The 1st input data_layout %s should be equal to 2ed input "
                 "data_layout %s.",
                 tensor1.data_layout(),
                 tensor2.data_layout());
      IR_ENFORCE(tensor1.lod() == tensor2.lod(),
                 "The 1st input lod %s should be equal to 2ed input lod %s.",
                 tensor1.lod(),
                 tensor2.lod());
      IR_ENFORCE(
          tensor1.offset() == tensor2.offset(),
          "The 1st input offset %s should be equal to 2ed input offset %s.",
          tensor1.offset(),
          tensor2.offset());
      IR_ENFORCE(
          tensor1.place() == tensor2.place(),
          "The 1st input place %s should be equal to 2ed input place %s.",
          tensor1.place(),
          tensor2.place());
    } else {
      IR_ENFORCE(out1 == out2,
                 "The 1st input type %s should be equal to 2ed input type %s.",
                 out1,
                 out2);
    }
  }
  VLOG(4) << "End Verifying for: AssignArray_Op.";
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::WhileOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::HasElementsOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AssertOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SelectInputOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SelectOutputOp)

#endif
