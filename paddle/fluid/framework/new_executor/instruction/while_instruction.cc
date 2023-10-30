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
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
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

  cond_var_ = inner_scope->GetVar(
      parent_exe_info->GetValue2VarName().at(while_op.operand_source(0)));
  for (size_t i = 1; i < while_op.num_operands(); ++i) {
    inputs_.push_back(inner_scope->GetVar(
        parent_exe_info->GetValue2VarName().at(while_op.operand_source(i))));
  }

  for (size_t i = 0; i < while_op.num_results(); ++i) {
    outputs_.push_back(inner_scope->GetVar(
        parent_exe_info->GetValue2VarName().at(while_op.result(i))));
  }

  body_block_ = while_op.body_block();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *parent_exe_info, &inputs);
  auto body_outside_inputs =
      GetOutsideOpInputs(body_block_, *parent_exe_info, &inputs);
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_NE(
          parent_exe_info->GetValue2VarName().find(value),
          parent_exe_info->GetValue2VarName().end(),
          phi::errors::PreconditionNotMet(
              "output should in name map, [%d] 'th output of [%s] op",
              i,
              "while op"));
      std::vector<int> outputs_id = GetValueIds(value, *parent_exe_info);
      outputs.emplace(value, outputs_id);
    }
  }
  SetOutputs(outputs);

  Scope* body_scope = &(parent_exe_info->GetScope()->NewScope());
  auto body_exe_info = parent_exe_info->NewChild(body_scope);
  for (size_t i = 0; i < body_block_->args_size(); ++i) {
    auto var_name = "body_block_arg_" + std::to_string(i);
    body_scope->Var(var_name);
    body_exe_info->Add(body_block_->argument(i), var_name);
  }
  body_inter_ = std::unique_ptr<PirInterpreter>(new PirInterpreter(
      place, {}, body_block_, body_scope, body_exe_info, {}));

  std::set<std::string> body_skip_gc_names_set;
  auto body_block_outputs = GetYiedOpInputs(body_block_);
  for (auto value : body_block_outputs) {
    body_outputs_.push_back(body_inter_->GetNameByValue(value));
    body_skip_gc_names_.push_back(body_inter_->GetNameByValue(value));
    body_skip_gc_names_set.insert(body_inter_->GetNameByValue(value));
  }
  for (auto value : body_outside_inputs) {
    body_skip_gc_names_.push_back(body_inter_->GetNameByValue(value));
    body_skip_gc_names_set.insert(body_inter_->GetNameByValue(value));
  }
  body_inter_->SetSkipGcVars(body_skip_gc_names_set);
}

void WhileInstruction::CopyInputsToOutputs() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    outputs_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
        inputs_[i]->Get<phi::DenseTensor>());
  }
}

void WhileInstruction::PassArgsToBodyBlock() {
  for (size_t i = 0; i < body_block_->args_size(); ++i) {
    auto block_arg = body_block_->argument(i);
    auto var_name = body_inter_->GetNameByValue(block_arg);
    auto* inner_var = body_inter_->local_scope()->GetVar(var_name);
    inner_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
        outputs_[i]->Get<phi::DenseTensor>());
  }
}

void WhileInstruction::GetValueFromBodyBlock() {
  cond_var_->GetMutable<phi::DenseTensor>()->ShareDataWith(
      body_inter_->local_scope()
          ->GetVar(body_outputs_[0])
          ->Get<phi::DenseTensor>());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    auto& out_var_name = body_outputs_[i + 1];
    auto* out_var = body_inter_->local_scope()->GetVar(out_var_name);
    outputs_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
        out_var->Get<phi::DenseTensor>());
  }
}
void WhileInstruction::Run() {
  CopyInputsToOutputs();
  while (cond_var_->Get<phi::DenseTensor>().data<bool>()[0]) {
    PassArgsToBodyBlock();
    body_inter_->Run({}, false);
    GetValueFromBodyBlock();
  }
}

}  // namespace framework
}  // namespace paddle
