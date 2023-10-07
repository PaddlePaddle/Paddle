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

#include "paddle/fluid/framework/new_executor/instruction/cond_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/new_ir_interpreter.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
namespace paddle {
namespace framework {

std::vector<pir::Value> GetYiedOpInputs(pir::Block* block) {
  std::vector<pir::Value> vec_res;
  for (auto op : (*block)) {
    if (op->name() == "cf.yield") {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        vec_res.push_back(op->operand_source(i));
      }
    }
  }

  return vec_res;
}

void GetInputIds(
    pir::Operation* op,
    Scope* inner_scope,
    const std::unordered_map<::pir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name,
    std::unordered_map<pir::Value, std::vector<int>>* input_ids) {
  for (size_t i = 0; i < op->num_operands(); i++) {
    pir::Value value = op->operand_source(i);
    if (value) {
      PADDLE_ENFORCE_NE(
          value_2_var_name.find(value),
          value_2_var_name.end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              "if op"));
      std::vector<int> inputs_id = GetValueIds(value,
                                               inner_scope,
                                               value_2_var_name,
                                               var_name_2_id,
                                               variable_2_var_name);
      input_ids->emplace(value, inputs_id);
    }
  }
}

void GetOutsideOpInputs(
    pir::Block* block,
    Scope* inner_scope,
    const std::unordered_map<::pir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name,
    std::unordered_map<pir::Value, std::vector<int>>* input_ids) {
  std::unordered_set<pir::Value> inner_outputs;
  for (auto op : (*block)) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      inner_outputs.insert(op->result(i));
    }
  }

  for (auto op : (*block)) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      pir::Value value = op->operand_source(i);
      if (value && (!inner_outputs.count(value))) {
        PADDLE_ENFORCE_NE(
            value_2_var_name.find(value),
            value_2_var_name.end(),
            phi::errors::PreconditionNotMet(
                "input should in name map, [%d] 'th input of [%s] op",
                i,
                "if op"));
        std::vector<int> inputs_id = GetValueIds(value,
                                                 inner_scope,
                                                 value_2_var_name,
                                                 var_name_2_id,
                                                 variable_2_var_name);

        input_ids->emplace(value, inputs_id);
      }
    }
  }
}

CondInstruction::CondInstruction(
    size_t id,
    const platform::Place& place,
    pir::Operation* op,
    Scope* scope,
    Scope* local_scope,
    ValueExecutionInfo* parent_exe_info,
    const std::map<pir::Block*, paddle::framework::Scope*>& sub_blocks)
    : InstructionBase(id, place) {
  op_ = op;
  VLOG(6) << "finish process dist attributes";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  Scope* inner_scope = local_scope == nullptr ? scope : local_scope;

  VLOG(6) << "finish process inputs outputs index";

  PADDLE_ENFORCE(
      op->isa<paddle::dialect::IfOp>(),
      phi::errors::PreconditionNotMet("Cond instruction only support if op"));

  auto if_op = op->dyn_cast<paddle::dialect::IfOp>();

  for (size_t i = 0; i < if_op.num_results(); ++i) {
    if_op_outputs_.push_back(inner_scope->GetVar(
        parent_exe_info->GetValue2VarName().at(if_op.result(i))));
  }

  auto cond_value = if_op.operand_source(0);
  auto var_name = parent_exe_info->GetValue2VarName().at(cond_value);
  cond_var = inner_scope->FindVar(var_name);

  auto true_branch_block = if_op.true_block();
  auto false_branch_block = if_op.false_block();

  auto true_branch_yied_inputs = GetYiedOpInputs(true_branch_block);
  auto false_branch_yied_inputs = GetYiedOpInputs(false_branch_block);

  auto true_scope = sub_blocks.at(true_branch_block);
  true_branch_inter =
      new NewIRInterpreter(place,
                           {},
                           true_branch_block,
                           true_scope,
                           parent_exe_info->NewChild(true_scope),
                           {});

  std::set<std::string> true_skip_gc_names_set;
  for (auto value : true_branch_yied_inputs) {
    true_skip_gc_names_.push_back(true_branch_inter->GetNameByValue(value));
    true_skip_gc_names_set.insert(true_branch_inter->GetNameByValue(value));
  }
  true_branch_inter->SetSkipGcVars(true_skip_gc_names_set);

  auto false_scope = sub_blocks.at(false_branch_block);
  false_branch_inter =
      new NewIRInterpreter(place,
                           {},
                           false_branch_block,
                           false_scope,
                           parent_exe_info->NewChild(false_scope),
                           {});

  std::set<std::string> false_skip_gc_names_set;
  for (auto value : false_branch_yied_inputs) {
    false_skip_gc_names_.push_back(false_branch_inter->GetNameByValue(value));
    false_skip_gc_names_set.insert(false_branch_inter->GetNameByValue(value));
  }
  false_branch_inter->SetSkipGcVars(false_skip_gc_names_set);

  // the true branch and false branch input will be the if_op inputs

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op,
              inner_scope,
              parent_exe_info->GetValue2VarName(),
              parent_exe_info->GetVarName2Id(),
              parent_exe_info->GetVar2VarName(),
              &inputs);
  GetOutsideOpInputs(true_branch_block,
                     inner_scope,
                     parent_exe_info->GetValue2VarName(),
                     parent_exe_info->GetVarName2Id(),
                     parent_exe_info->GetVar2VarName(),
                     &inputs);

  GetOutsideOpInputs(false_branch_block,
                     inner_scope,
                     parent_exe_info->GetValue2VarName(),
                     parent_exe_info->GetVarName2Id(),
                     parent_exe_info->GetVar2VarName(),
                     &inputs);
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_NE(
          parent_exe_info->GetValue2VarName().find(value),
          parent_exe_info->GetValue2VarName().end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              "if op"));
      std::vector<int> outputs_id =
          GetValueIds(value,
                      inner_scope,
                      parent_exe_info->GetValue2VarName(),
                      parent_exe_info->GetVarName2Id(),
                      parent_exe_info->GetVar2VarName());
      outputs.emplace(value, outputs_id);
    }
  }
  SetOutputs(outputs);
}

void CondInstruction::CopyBranchOutput(
    const std::vector<std::string>& var_names, const NewIRInterpreter* inter) {
  for (size_t i = 0; i < var_names.size(); ++i) {
    auto* inner_var = inter->local_scope()->GetVar(var_names[i]);

    if_op_outputs_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
        inner_var->Get<phi::DenseTensor>());
  }
}

void CondInstruction::Run() {
  if (cond_var->Get<phi::DenseTensor>().data<bool>()[0]) {
    true_branch_inter->Run({}, false);
    CopyBranchOutput(true_skip_gc_names_, true_branch_inter);
  } else {
    false_branch_inter->Run({}, false);
    CopyBranchOutput(false_skip_gc_names_, false_branch_inter);
  }

  // copy ouptut
}

}  // namespace framework
}  // namespace paddle
