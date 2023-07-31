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

#include "paddle/fluid/ir_adaptor/translator/program_translator.h"

#include <unordered_map>

#include "glog/logging.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/ir_adaptor/translator/op_translator.h"
#include "paddle/fluid/ir_adaptor/translator/type_translator.h"
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace translator {

using ProgramDesc = ::paddle::framework::ProgramDesc;
using BlockDesc = ::paddle::framework::BlockDesc;
using VarDesc = ::paddle::framework::VarDesc;

const std::unordered_set<std::string> ProgramTranslator::no_cast_var_names = {
    "feed",
    "fetch",
};

ProgramTranslator::ProgramTranslator(const ProgramDesc* legacy_program,
                                     ir::Program* program)
    : legacy_program_(legacy_program), program_(program) {
  ctx_ = ir::IrContext::Instance();
}

void ProgramTranslator::Translate() {
  PADDLE_ENFORCE_EQ(
      legacy_program_->Size(),
      1u,
      platform::errors::PreconditionNotMet(
          "Not support multi block ProgramDesc translated, now has %d blocks",
          legacy_program_->Size()));
  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    GetParameterForSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    InsertOperationToSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    SetParameterFromSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    SetStopGradientAttributeForAllValue(block);
  }
}

inline ir::Operation* InsertGetParamaterOp(ir::IrContext* ctx,
                                           const VarDesc* var) {
  auto& type_translator = TypeTranslator::instance();
  std::string get_parameter_op_name(ir::GetParameterOp::name());
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(get_parameter_op_name);
  std::unordered_map<std::string, ir::Attribute> op_attribute_map = {
      {"parameter_name", ir::StrAttribute::get(ctx, var->Name())},
  };

  ir::Type translated_var_type = type_translator[var->GetType()](ctx, *var);
  ir::Operation* operation = ir::Operation::Create(
      {}, op_attribute_map, {translated_var_type}, op_info);
  return operation;
}

inline ir::Operation* InsertSetParamaterOp(ir::IrContext* ctx,
                                           ir::OpResult defining_op_result,
                                           const VarDesc* var) {
  std::string set_parameter_op_name(ir::SetParameterOp::name());
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(set_parameter_op_name);
  std::unordered_map<std::string, ir::Attribute> op_attribute_map = {
      {"parameter_name", ir::StrAttribute::get(ctx, var->Name())},
  };

  ir::Operation* operation = ir::Operation::Create(
      {defining_op_result}, op_attribute_map, {}, op_info);
  return operation;
}

void ProgramTranslator::GetParameterForSingleBlock(const BlockDesc& block) {
  for (auto& var : block.AllVars()) {
    if (!var->Persistable()) continue;
    if (param_map_.count(var->Name()) != 0) continue;
    if (no_cast_var_names.count(var->Name()) != 0) continue;

    parameter_name_mappings_[var->Name()] = var;
  }

  std::unordered_set<std::string> inner_defining_variables;

  for (auto op_desc : block.AllOps()) {
    for (const auto& n : op_desc->Inputs()) {
      const auto& input_var_names = n.second;
      for (const auto& var_name : input_var_names) {
        if (no_cast_var_names.count(var_name) != 0) continue;
        VarDesc* var_desc = nullptr;

        bool is_parameter = (parameter_name_mappings_.find(var_name) !=
                             parameter_name_mappings_.end());
        is_parameter &= (parameter_visited_.count(var_name) == 0);
        if (is_parameter) {
          var_desc = parameter_name_mappings_[var_name];
        }
        bool is_unseen_variable =
            (inner_defining_variables.count(var_name) == 0);
        if (is_unseen_variable) {
          var_desc = block.FindVarRecursive(var_name);
        }

        bool need_get_parameter_op = is_parameter || is_unseen_variable;
        if (need_get_parameter_op) {
          ir::Operation* op = InsertGetParamaterOp(ctx_, var_desc);
          program_->block()->push_back(op);
          param_map_[var_name] = VariableDefiningInfo(op->result(0));
          VLOG(10) << "[op translated][get parameter]" << op;

          program_->SetParameter(var_name, nullptr);
          parameter_visited_.insert(var_name);
          inner_defining_variables.insert(var_name);
        }
      }
    }
    for (const auto& n : op_desc->Outputs()) {
      const auto& output_var_names = n.second;
      for (const auto& var_name : output_var_names) {
        inner_defining_variables.insert(var_name);
      }
    }
  }
}

void ProgramTranslator::InsertOperationToSingleBlock(const BlockDesc& block) {
  auto& op_translator = OpTranslator::instance();
  for (auto op : block.AllOps()) {
    OpTranslateFn& fn = op_translator[op->Type()];
    if (op->Type() == "shaddow_output") {
      if (!param_map_.count(op->Input("x")[0])) {
        continue;
      }
    }
    ir::Operation* operation = fn(ctx_, &param_map_, *op, program_);
    VLOG(10) << "[op translated][special]" << operation;
  }
}

void ProgramTranslator::SetParameterFromSingleBlock(const BlockDesc& block) {
  const auto& ops = block.AllOps();
  for (auto op_desc = ops.rbegin(); op_desc != ops.rend(); op_desc++) {
    for (const auto& n : (*op_desc)->Outputs()) {
      const auto& output_var_names = n.second;
      for (const auto& var_name : output_var_names) {
        bool need_set_parameter_op = (parameter_name_mappings_.find(var_name) !=
                                      parameter_name_mappings_.end());
        need_set_parameter_op &= (parameter_visited_.count(var_name) == 0);
        need_set_parameter_op &= (param_map_.count(var_name) != 0);
        if (need_set_parameter_op) {
          ir::OpResult defining_op_result = param_map_[var_name].value;
          if (!defining_op_result) {
            continue;
          }

          ir::Operation* op = InsertSetParamaterOp(
              ctx_, defining_op_result, parameter_name_mappings_[var_name]);

          ir::Block* block = program_->block();
          ir::Block::iterator insert_pos = std::find(
              block->begin(), block->end(), defining_op_result.owner());

          IR_ENFORCE(
              insert_pos != block->end(),
              "Parameter %s must have corresponding its defining operation",
              var_name);
          insert_pos++;

          block->insert(insert_pos, op);
          VLOG(10) << "[op translated][set parameter]" << op;

          program_->SetParameter(var_name, nullptr);
          parameter_visited_.insert(var_name);
        }
      }
    }
  }
}

void ProgramTranslator::SetStopGradientAttributeForAllValue(
    const BlockDesc& block) {
  // Currently we set stop gradient for operation that generated a value
  // connected with VarDesc
  for (const auto& [var_name, value_info] : param_map_) {
    VLOG(10) << "[op translated][stop gradient]" << var_name;
    VarDesc* var = block.FindVarRecursive(var_name);
    if (var == nullptr) {
      continue;
    }
    ir::OpResult value = value_info.value;
    if (!value) {
      PADDLE_THROW(phi::errors::PreconditionNotMet(
          "Value of [%s] can not ber None", var_name));
    }
    auto* defining_op = value.owner();
    PADDLE_ENFORCE_NOT_NULL(
        defining_op,
        phi::errors::PreconditionNotMet(
            "Defining operator of [%s] can not be nullptr", var_name));
    VLOG(8) << "[op translated][stop gradient]" << var_name
            << " from: " << defining_op->name();
    std::vector<ir::Attribute> stop_gradients;
    if (defining_op->HasAttribute(kAttrStopGradients)) {
      stop_gradients = defining_op->attribute(kAttrStopGradients)
                           .dyn_cast<ir::ArrayAttribute>()
                           .AsVector();
    } else {
      stop_gradients = std::vector<ir::Attribute>(
          defining_op->num_results(), ir::BoolAttribute::get(ctx_, false));
    }
    stop_gradients[value.GetResultIndex()] =
        ir::BoolAttribute::get(ctx_, var->StopGradient());
    defining_op->set_attribute(kAttrStopGradients,
                               ir::ArrayAttribute::get(ctx_, stop_gradients));
  }
}

}  // namespace translator
}  // namespace paddle
