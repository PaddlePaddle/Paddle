// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
paddle::dialect::PyLayerOp
#else

#include <unordered_map>

#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace paddle {
namespace dialect {

std::unordered_set<pir::Value> GetInternalInputs(pir::Block *block) {
  std::unordered_set<pir::Value> inner_inputs;
  for (auto &op : *block) {
    std::string op_name = op.name();
    if (op.attributes().count("op_name")) {
      op_name = op.attributes()
                    .at("op_name")
                    .dyn_cast<pir::StrAttribute>()
                    .AsString();
    }
    VLOG(8) << "GetInternalInputs of " << op_name;
    if (op.num_regions()) {
      for (size_t i = 0; i < op.num_regions(); ++i) {
        for (auto &sub_block : op.region(i)) {
          std::unordered_set<pir::Value> sub_set =
              GetInternalInputs(&sub_block);
          inner_inputs.insert(sub_set.begin(), sub_set.end());
        }
      }
    }
    if (op.isa<pir::TuplePopOp>()) {
      auto tuple_pop_op = op.dyn_cast<pir::TuplePopOp>();
      if (tuple_pop_op.has_container()) {
        inner_inputs.insert(tuple_pop_op.container());
      }
    }
    for (size_t i = 0; i < op.num_operands(); ++i) {
      inner_inputs.insert(op.operand_source(i));
      VLOG(10) << op_name << "'s inner_input: " << op.operand_source(i).impl();
    }
  }
  return inner_inputs;
}

const char *PyLayerOp::attributes_name[1] = {
    kBackwardFunctionIdAttrName};  // NOLINT

void PyLayerOp::Build(pir::Builder &builder,             // NOLINT
                      pir::OperationArgument &argument,  // NOLINT
                      const std::vector<pir::Value> &inputs,
                      std::vector<pir::Type> &&output_types,
                      int backward_function_id) {
  VLOG(4) << "Start building PyLayerOp";
  argument.AddAttribute(
      kBackwardFunctionIdAttrName,
      pir::Int32Attribute::get(builder.ir_context(), backward_function_id));

  argument.AddInputs(inputs);
  argument.output_types.swap(output_types);
  argument.AddRegion().emplace_back();
  VLOG(4) << "Finish building PyLayerOp";
}

void PyLayerOp::Build(pir::Builder &builder,             // NOLINT
                      pir::OperationArgument &argument,  // NOLINT
                      const std::vector<pir::Value> &inputs,
                      std::unique_ptr<pir::Block> &&fwd_block,
                      int backward_function_id) {
  VLOG(4) << "Start build PyLayerOp";

  PADDLE_ENFORCE_NOT_NULL(
      fwd_block,
      common::errors::InvalidArgument("The sub-block for building pylayer_op "
                                      "can't be None"));

  PADDLE_ENFORCE_NE(
      fwd_block->empty(),
      true,
      common::errors::InvalidArgument("The sub-block for building pylayer_op "
                                      "can't be empty"));

  PADDLE_ENFORCE_EQ(fwd_block->back().isa<pir::YieldOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The last op of sub-block for building pylayer_op "
                        "must be pir::YieldOp"));

  auto &op = fwd_block->back();

  auto outs_stop_gradient_attr = true;
  for (size_t i = 0; i < op.num_operands(); ++i) {
    argument.AddOutput(op.operand(i).type());
    auto bool_attr = op.operand_source(i).attribute<pir::BoolAttribute>(
        pir::kStopGradientAttrName);
    if (!bool_attr || (bool_attr && !bool_attr.data())) {
      outs_stop_gradient_attr = false;
    }
  }
  std::vector<pir::Attribute> outs_stop_gradient(
      op.num_operands(), builder.bool_attr(outs_stop_gradient_attr));

  argument.AddAttribute(
      kBackwardFunctionIdAttrName,
      pir::Int32Attribute::get(builder.ir_context(), backward_function_id));
  argument.AddAttribute(
      pir::kStopGradientAttrName,
      pir::ArrayAttribute::get(builder.ir_context(), outs_stop_gradient));

  argument.AddRegion().push_back(fwd_block.release());
  argument.AddInputs(inputs);
}

pir::Block &PyLayerOp::forward_block() {
  pir::Region &region = forward_region();
  if (region.empty()) {
    region.emplace_back();
  }

  return region.front();
}

void PyLayerOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(*op);
  os << " = ";
  printer.PrintOpName(*op);
  printer.PrintOpId(*op);

  printer.PrintOpOperands(*op);
  printer.PrintAttributeMap(*op);
  os << " -> ";
  printer.PrintOpReturnType(*op);
  os << " {\n";
  printer.AddIndentation();
  for (auto &item : forward_block()) {
    printer.PrintOperation(item);
    os << "\n";
  }
  printer.DecreaseIndentation();
  os << printer.indentation() << "}";
}

void PyLayerOp::VerifySig() {
  VLOG(4) << "Start Verifying attributes for: PyLayerOp.";
  auto &attributes = this->attributes();
  PADDLE_ENFORCE_GT(
      attributes.count(kBackwardFunctionIdAttrName),
      0,
      common::errors::InvalidArgument("backward_function_id does not exist."));
  PADDLE_ENFORCE_EQ(
      attributes.at(kBackwardFunctionIdAttrName).isa<pir::Int32Attribute>(),
      true,
      common::errors::InvalidArgument(
          "Type of attribute: value is not pir::Int32Attribute."));
  VLOG(4) << "Finish Verifying attributes for: PyLayerOp.";
}

void PyLayerOp::VerifyRegion() {
  VLOG(4) << "Start Verifying sub regions for: PyLayerOp.";
  VLOG(4) << "Start Verifying forward block.";
  PADDLE_ENFORCE_EQ((*this)->region(0).size(),
                    1u,
                    common::errors::PreconditionNotMet(
                        "The size %d of forward_region must be 1.",
                        (*this)->region(0).size()));
  if ((*this)->num_results() != 0) {
    auto &fwd_last_op = (*this)->region(0).front().back();
    PADDLE_ENFORCE_EQ(true,
                      fwd_last_op.isa<pir::YieldOp>(),
                      common::errors::PreconditionNotMet(
                          "The last of forward block must be YieldOp"));
    PADDLE_ENFORCE_EQ(
        fwd_last_op.num_operands(),
        (*this)->num_results(),
        common::errors::PreconditionNotMet(
            "The size of last of forward block op's input must be "
            "equal to PyLayerOp's outputs num."));
  }
}

void PyLayerOp::UpdateOutput() {
  PADDLE_ENFORCE_NOT_NULL(*this,
                          common::errors::InvalidArgument(
                              "The pylayer_op in PyLayerOp used to update "
                              "output can't be nullptr"));
  auto block = parent();
  PADDLE_ENFORCE_NOT_NULL(
      block,
      common::errors::InvalidArgument(
          "The parent block of pylayer_op which used to update "
          "output can't be nullptr"));
  pir::Block::Iterator iter = **this;
  pir::Builder builder(ir_context(), false);
  auto new_pylayer_op = builder.Build<PyLayerOp>(
      inputs(), forward_region().TakeBack(), backward_function_id());
  block->Assign(iter, new_pylayer_op);
  PyLayerOp::operator=(new_pylayer_op);
  VerifyRegion();
}

PyLayerOp PyLayerOp::UpdateInput() {
  PADDLE_ENFORCE_NOT_NULL(*this,
                          common::errors::InvalidArgument(
                              "The pylayer_op in PyLayerOp used to update "
                              "output can't be nullptr"));
  auto program_block = parent();
  PADDLE_ENFORCE_NOT_NULL(
      program_block,
      common::errors::InvalidArgument(
          "The parent block of pylayer_op which used to update "
          "output can't be nullptr"));

  pir::Block &block = forward_block();
  std::vector<pir::Value> input_values = inputs();

  std::unordered_set<pir::Value> inner_inputs;
  inner_inputs = GetInternalInputs(&block);

  for (size_t arg_id = 0; arg_id < block.args_size();) {
    if (block.arg(arg_id) && (!inner_inputs.count(block.arg(arg_id)))) {
      block.EraseArg(arg_id);
      continue;
    }
    ++arg_id;
  }

  bool need_build_new_pylayer = false;
  std::vector<pir::Value> new_pylayer_inputs;

  for (auto value : input_values) {
    if (value && (!inner_inputs.count(value))) {
      need_build_new_pylayer = true;
      continue;
    }
    new_pylayer_inputs.push_back(value);
  }

  if (need_build_new_pylayer) {
    ::pir::IrContext *ctx = ::pir::IrContext::Instance();

    ::pir::Builder builder = ::pir::Builder(ctx, program_block);
    builder.set_insertion_point(&(**this));
    auto new_pylayer = builder.Build<PyLayerOp>(new_pylayer_inputs,
                                                forward_region().TakeBack(),
                                                backward_function_id());
    (**this).ReplaceAllUsesWith(new_pylayer.outputs());
    pir::Block::Iterator iter = **this;
    iter = program_block->erase(iter);
    return new_pylayer;
  }
  return *this;
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PyLayerOp)

#endif
