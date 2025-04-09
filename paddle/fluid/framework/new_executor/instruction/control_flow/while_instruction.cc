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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/while_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

namespace paddle::framework {

WhileInstruction::WhileInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    ValueExecutionInfo* parent_exe_info,
    interpreter::ExecutionConfig execution_config)
    : InstructionBase(id, place),
      inputs_(),
      outputs_(),
      body_inter_(nullptr),
      external_input_names_() {
  PADDLE_ENFORCE(op->isa<paddle::dialect::WhileOp>(),
                 common::errors::PreconditionNotMet(
                     "While instruction only support While op"));
  op_ = op;
  auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();
  body_block_ = &while_op.body();

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  cond_var_ = parent_exe_info->GetVarByValue(while_op.cond());
  for (size_t i = 1; i < while_op.num_operands(); ++i) {
    inputs_.push_back(
        parent_exe_info->GetVarByValue(while_op.operand_source(i)));
  }
  for (size_t i = 0; i < while_op.num_results(); ++i) {
    outputs_.push_back(parent_exe_info->GetVarByValue(while_op.result(i)));
  }

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *parent_exe_info, &inputs);
  auto body_outside_inputs =
      GetExternalInputs(body_block_, *parent_exe_info, &inputs);
  // NOTE(chenxi67): the variable corresponding to container value if a
  // <VariableRefArray> Type. It will recursively get the ID of internal
  // variables when use GetValueId() method. However, the copy_var pushed into
  // the tuple does not have a corresponding ID, and will insert a -1. Here we
  // remove the value of -1.
  for (auto& item : inputs) {
    auto& var_vec = item.second;
    for (auto it = var_vec.begin(); it != var_vec.end();) {
      if (*it == -1) {
        it = var_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_NE(
          parent_exe_info->GetValue2VarName().find(value),
          parent_exe_info->GetValue2VarName().end(),
          common::errors::PreconditionNotMet(
              "output should in name map, [%d] 'th output of [%s] op",
              i,
              "while op"));
      std::vector<int> outputs_id = GetValueIds(value, *parent_exe_info);
      outputs.emplace(value, outputs_id);
    }
  }
  InsertTuplePushContainerToOuts(body_block_, *parent_exe_info, &outputs);
  InsertInplacedExternalInputsToOuts(
      body_block_, body_outside_inputs, *parent_exe_info, &outputs);
  SetOutputs(outputs);

  Scope* body_scope = &(parent_exe_info->GetScope()->NewScope());
  auto body_exe_info = parent_exe_info->NewChild(body_scope);
  for (size_t i = 0; i < body_block_->args_size(); ++i) {
    std::stringstream ss;
    ss << this
       << std::chrono::high_resolution_clock::now().time_since_epoch().count()
       << "body_block_arg_";
    auto var_name = ss.str() + std::to_string(i);
    body_scope->Var(var_name);
    body_exe_info->Add(body_block_->arg(i), var_name);
  }
  auto skip_gc_vars = execution_config.skip_gc_vars;
  execution_config.skip_gc_vars.clear();
  execution_config.create_local_scope = true;
  body_inter_ = std::unique_ptr<PirInterpreter>(new PirInterpreter(
      place, {}, body_block_, body_scope, body_exe_info, execution_config));

  if (body_block_->back().isa<pir::YieldOp>()) {
    const auto& op = body_block_->back();
    inner_cond_ = body_inter_->GetNameByValue(op.operand_source(0));
    skip_gc_vars.insert(inner_cond_);
  }
  for (auto value : body_outside_inputs) {
    auto name = body_inter_->GetNameByValue(value);
    external_input_names_.insert(name);
    skip_gc_vars.insert(name);
  }
  body_inter_->SetSkipGcVars(skip_gc_vars);

  if (VLOG_IS_ON(6)) {
    std::stringstream body_skip_gc_names;
    for (const auto& var_name : skip_gc_vars) {
      body_skip_gc_names << " " << var_name;
    }
    VLOG(6) << "body_skip_gc_names include: " << body_skip_gc_names.str();
  }
}

void WhileInstruction::ShareInputsToOutputs() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (inputs_[i]->IsType<phi::DenseTensor>()) {
      outputs_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
          inputs_[i]->Get<phi::DenseTensor>());
    } else if (inputs_[i]->IsType<phi::TensorArray>()) {
      const auto& input_array = inputs_[i]->Get<phi::TensorArray>();
      auto* output_array = outputs_[i]->GetMutable<phi::TensorArray>();
      *output_array = input_array;
    } else {
      PADDLE_THROW(common::errors::Unimplemented("unsupported type %d",
                                                 inputs_[i]->Type()));
    }
  }
}

void WhileInstruction::ShareOutputsToBlockArgs() {
  for (size_t i = 0; i < body_block_->args_size(); ++i) {
    auto block_arg = body_block_->arg(i);
    auto var_name = body_inter_->GetNameByValue(block_arg);
    auto* inner_var = body_inter_->local_scope()->GetVar(var_name);

    if (outputs_[i]->IsType<phi::DenseTensor>()) {
      inner_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
          outputs_[i]->Get<phi::DenseTensor>());
    } else if (outputs_[i]->IsType<phi::TensorArray>()) {
      const auto& outer_array = outputs_[i]->Get<phi::TensorArray>();
      auto* inner_array = inner_var->GetMutable<phi::TensorArray>();
      *inner_array = outer_array;
      VLOG(10) << inner_var
               << " should be created: " << inner_var->IsInitialized();
    } else {
      PADDLE_THROW(common::errors::Unimplemented("unsupported type %d",
                                                 inner_var->Type()));
    }
  }
}

void WhileInstruction::ShareConditionData() {
  auto inner_cond_var = body_inter_->local_scope()->GetVar(inner_cond_);
  cond_var_->GetMutable<phi::DenseTensor>()->ShareDataWith(
      inner_cond_var->Get<phi::DenseTensor>());
}

void WhileInstruction::SetOutputHooks(
    const std::vector<PirHookFunc>& hookfuncs) {
  body_inter_->SetOutputHooks(hookfuncs);
}

void WhileInstruction::SetInputHooks(
    const std::vector<PirHookFunc>& hookfuncs) {
  body_inter_->SetInputHooks(hookfuncs);
}

void WhileInstruction::CheckGCEarly(const CheckGCEarlyHook& check_gc_early) {
  check_gc_early_ = check_gc_early;
}

void WhileInstruction::Run() {
#ifdef PADDLE_WITH_DNNL
  // Executor on being destroyed clears oneDNN cache and resets
  // registered model data layout. This is unwanted for nested
  // Executors (executors declared inside control ops)
  paddle::platform::DontClearMKLDNNCache(body_inter_->GetPlace());
#endif
  ShareInputsToOutputs();

  if (check_gc_early_) {
    check_gc_early_(this);
  }

  VLOG(6) << "while instruction start loop ...";
  while (GetCondData(cond_var_->Get<phi::DenseTensor>())) {
    VLOG(6) << "while instruction pass args to body block";
    ShareOutputsToBlockArgs();
    VLOG(6) << "while instruction interpretercore run";
    body_inter_->Run({}, false);
    VLOG(6) << "while instruction get condition value form body block";
    ShareConditionData();
  }
  VLOG(6) << "while instruction run done";
}

}  // namespace paddle::framework
