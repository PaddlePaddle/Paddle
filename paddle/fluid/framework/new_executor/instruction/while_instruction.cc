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

#include "paddle/fluid/framework/new_executor/instruction/while_instruction.h"

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
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"

namespace paddle {
namespace framework {

WhileInstruction::WhileInstruction(size_t id,
                                   const platform::Place& place,
                                   pir::Operation* op,
                                   Scope* scope,
                                   Scope* local_scope,
                                   ValueExecutionInfo* parent_exe_info)
    : InstructionBase(id, place) {
  op_ = op;
  VLOG(6) << "finish process dist attributes";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  Scope* inner_scope = local_scope == nullptr ? scope : local_scope;

  VLOG(6) << "finish process inputs outputs index";

  PADDLE_ENFORCE(op->isa<paddle::dialect::WhileOp>(),
                 phi::errors::PreconditionNotMet(
                     "While instruction only support While op"));

  auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();

  for (size_t i = 0; i < while_op.num_operands(); ++i) {
    inputs_.push_back(inner_scope->GetVar(
        parent_exe_info->GetValue2VarName().at(while_op.operand_source(i))));
  }

  for (size_t i = 0; i < while_op.num_results(); ++i) {
    outputs_.push_back(inner_scope->GetVar(
        parent_exe_info->GetValue2VarName().at(while_op.result(i))));
  }

  cond_block_ = while_op.cond_block();
  body_block_ = while_op.body_block();

  auto cond_yied_inputs = GetYiedOpInputs(cond_block_);
  auto body_yied_inputs = GetYiedOpInputs(body_block_);

  Scope* cond_scope = &(parent_exe_info->GetScope()->NewScope());
  auto cond_exe_info = parent_exe_info->NewChild(cond_scope);
  for (size_t i = 0; i < cond_block_->args_size(); ++i) {
    auto var_name = "block_arg_" + std::to_string(i);
    cond_scope->Var(var_name);
    cond_exe_info->Add(cond_block_->argument(i), var_name);
  }
  cond_inter_ = std::unique_ptr<NewIRInterpreter>(new NewIRInterpreter(
      place, {}, cond_block_, cond_scope, cond_exe_info, {}));

  std::set<std::string> cond_skip_gc_names_set;
  for (auto value : cond_yied_inputs) {
    cond_skip_gc_names_.push_back(cond_inter_->GetNameByValue(value));
    cond_skip_gc_names_set.insert(cond_inter_->GetNameByValue(value));
  }
  cond_inter_->SetSkipGcVars(cond_skip_gc_names_set);

  auto cond_value = cond_yied_inputs[0];
  auto var_name = cond_inter_->GetNameByValue(cond_value);
  cond_var = cond_inter_->local_scope()->GetVar(var_name);

  Scope* body_scope = &(parent_exe_info->GetScope()->NewScope());
  auto body_exe_info = parent_exe_info->NewChild(body_scope);
  for (size_t i = 0; i < body_block_->args_size(); ++i) {
    auto var_name = "body_block_arg_" + std::to_string(i);
    body_scope->Var(var_name);
    body_exe_info->Add(body_block_->argument(i), var_name);
  }
  body_inter_ = std::unique_ptr<NewIRInterpreter>(new NewIRInterpreter(
      place, {}, body_block_, body_scope, body_exe_info, {}));

  std::set<std::string> body_skip_gc_names_set;
  for (auto value : body_yied_inputs) {
    body_skip_gc_names_.push_back(body_inter_->GetNameByValue(value));
    body_skip_gc_names_set.insert(body_inter_->GetNameByValue(value));
  }
  body_inter_->SetSkipGcVars(body_skip_gc_names_set);

  // the cond block and body block input also be the while_op inputs

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *parent_exe_info, &inputs);

  // TODO(phlrain): process cond and body block
  // GetOutsideOpInputs(cond_block_,
  //                    inner_scope,
  //                    parent_exe_info->GetValue2VarName(),
  //                    parent_exe_info->GetVarName2Id(),
  //                    parent_exe_info->GetVar2VarName(),
  //                    &inputs);

  // GetOutsideOpInputs(body_block_,
  //                    inner_scope,
  //                    parent_exe_info->GetValue2VarName(),
  //                    parent_exe_info->GetVarName2Id(),
  //                    parent_exe_info->GetVar2VarName(),
  //                    &inputs);
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
      std::vector<int> outputs_id = GetValueIds(value, *parent_exe_info);
      outputs.emplace(value, outputs_id);
    }
  }
  SetOutputs(outputs);
}

void WhileInstruction::CopyStepOutput() {
  for (size_t i = 0; i < body_skip_gc_names_.size(); ++i) {
    auto* inner_var =
        body_inter_->local_scope()->GetVar(body_skip_gc_names_[i]);

    outputs_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
        inner_var->Get<phi::DenseTensor>());
  }
}

void WhileInstruction::CopyWhileInputToBlockArgs(const NewIRInterpreter* inter,
                                                 ::pir::Block* block) {
  for (size_t i = 0; i < block->args_size(); ++i) {
    auto block_arg = block->argument(i);
    auto var_name = inter->GetNameByValue(block_arg);
    auto* inner_var = inter->local_scope()->GetVar(var_name);
    inner_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
        inputs_[i]->Get<phi::DenseTensor>());
  }
}

void WhileInstruction::CopyStepOutputToBlockArgs(const NewIRInterpreter* inter,
                                                 ::pir::Block* block) {
  for (size_t i = 0; i < block->args_size(); ++i) {
    auto& out_var_name = body_skip_gc_names_[i];
    auto* out_var = body_inter_->local_scope()->GetVar(out_var_name);

    auto block_arg = block->argument(i);
    auto block_in_var_name = inter->GetNameByValue(block_arg);

    auto* inner_var = inter->local_scope()->GetVar(block_in_var_name);

    inner_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
        out_var->Get<phi::DenseTensor>());
  }
}

void WhileInstruction::Run() {
  CopyWhileInputToBlockArgs(cond_inter_.get(), cond_block_);
  CopyWhileInputToBlockArgs(body_inter_.get(), body_block_);

  while (true) {
    cond_inter_->Run({}, false);

    if (cond_var->Get<phi::DenseTensor>().data<bool>()[0]) {
      body_inter_->Run({}, false);

      CopyStepOutputToBlockArgs(cond_inter_.get(), cond_block_);
      CopyStepOutputToBlockArgs(body_inter_.get(), body_block_);
    } else {
      break;
    }
  }

  // copy  output
  CopyStepOutput();
}

}  // namespace framework
}  // namespace paddle
