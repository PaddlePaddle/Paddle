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

ProgramTranslator::ProgramTranslator(const ProgramDesc* legacy_program,
                                     ir::Program* program)
    : legacy_program(legacy_program), program(program) {
  ctx = ir::IrContext::Instance();
}

const std::unordered_set<std::string> ProgramTranslator::no_cast_var_names = {
    "feed",
    "fetch",
};

void ProgramTranslator::Translate() {
  PADDLE_ENFORCE_EQ(
      legacy_program->Size(),
      1u,
      platform::errors::PreconditionNotMet(
          "Not support multi block ProgramDesc translated, now has %d blocks",
          legacy_program->Size()));

  for (size_t block_idx = 0; block_idx < legacy_program->Size(); block_idx++) {
    const BlockDesc& block = legacy_program->Block(block_idx);
    GetParameterForSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program->Size(); block_idx++) {
    const BlockDesc& block = legacy_program->Block(block_idx);
    InsertOperationToSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program->Size(); block_idx++) {
    const BlockDesc& block = legacy_program->Block(block_idx);
    SetParameterFromSingleBlock(block);
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
    if (param_map.count(var->Name()) != 0) continue;
    if (no_cast_var_names.count(var->Name()) != 0) continue;

    parameter_name_mappings[var->Name()] = var;
  }

  std::unordered_set<std::string> inner_defining_variables;

  for (auto op_desc : block.AllOps()) {
    for (const auto& n : op_desc->Inputs()) {
      const auto& input_var_names = n.second;
      for (const auto& var_name : input_var_names) {
        VarDesc* var_desc = nullptr;

        bool is_parameter = (parameter_name_mappings.find(var_name) !=
                             parameter_name_mappings.end());
        is_parameter &= (parameter_visited.count(var_name) == 0);
        if (is_parameter) {
          var_desc = parameter_name_mappings[var_name];
        }
        bool is_unseen_variable =
            (inner_defining_variables.count(var_name) == 0);
        if (is_unseen_variable) {
          var_desc = block.FindVarRecursive(var_name);
        }

        bool need_get_parameter_op = is_parameter || is_unseen_variable;
        if (need_get_parameter_op) {
          ir::Operation* op = InsertGetParamaterOp(ctx, var_desc);
          program->block()->push_back(op);
          param_map[var_name] = VariableDefiningInfo(op->result(0));
          VLOG(10) << "[op translated][get parameter]" << op;

          program->SetParameter(var_name, nullptr);
          parameter_visited.insert(var_name);
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
    ir::Operation* operation = fn(ctx, &param_map, program, *op);
    VLOG(10) << "[op translated][special]" << operation;
  }
}

void ProgramTranslator::SetParameterFromSingleBlock(const BlockDesc& block) {
  const auto& ops = block.AllOps();
  for (auto op_desc = ops.rbegin(); op_desc != ops.rend(); op_desc++) {
    for (const auto& n : (*op_desc)->Outputs()) {
      const auto& output_var_names = n.second;
      for (const auto& var_name : output_var_names) {
        bool need_set_parameter_op = (parameter_name_mappings.find(var_name) !=
                                      parameter_name_mappings.end());
        need_set_parameter_op &= (parameter_visited.count(var_name) == 0);
        if (need_set_parameter_op) {
          ir::OpResult defining_op_result = param_map[var_name].value;
          ir::Operation* op = InsertSetParamaterOp(
              ctx, defining_op_result, parameter_name_mappings[var_name]);

          ir::Block* block = program->block();
          ir::Block::iterator insert_pos = std::find(
              block->begin(), block->end(), defining_op_result.owner());

          IR_ENFORCE(
              insert_pos != block->end(),
              "Parameter %s must have corresponding its defining operation",
              var_name);
          insert_pos++;

          block->insert(insert_pos, op);
          VLOG(10) << "[op translated][set parameter]" << op;

          program->SetParameter(var_name, nullptr);
          parameter_visited.insert(var_name);
        }
      }
    }
  }
}

}  // namespace translator
}  // namespace paddle
