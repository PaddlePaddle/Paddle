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
                                   ValueExecutionInfo* parent_exe_info,
                                   const std::set<std::string>& skip_gc_vars)
    : InstructionBase(id, place) {
  op_ = op;
  VLOG(6) << "finish process dist attributes";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  VLOG(6) << "finish process inputs outputs index";

  PADDLE_ENFORCE(op->isa<paddle::dialect::WhileOp>(),
                 phi::errors::PreconditionNotMet(
                     "While instruction only support While op"));

  auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();

  cond_var_ = parent_exe_info->GetVarByValue(while_op.operand_source(0));

  for (size_t i = 1; i < while_op.num_operands(); ++i) {
    inputs_.push_back(
        parent_exe_info->GetVarByValue(while_op.operand_source(i)));
  }

  for (size_t i = 0; i < while_op.num_results(); ++i) {
    outputs_.push_back(parent_exe_info->GetVarByValue(while_op.result(i)));
  }

  body_block_ = &while_op.body();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *parent_exe_info, &inputs);
  auto body_outside_inputs =
      GetExternalInputs(body_block_, *parent_exe_info, &inputs);
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
    InsertTuplePushContinerToOuts(body_block_, *parent_exe_info, &outputs);
  }
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
  for (auto var_name : skip_gc_vars) {
    body_skip_gc_names_.push_back(var_name);
    body_skip_gc_names_set.insert(var_name);
  }
  body_inter_->SetSkipGcVars(body_skip_gc_names_set);

  if (VLOG_IS_ON(6)) {
    std::stringstream body_outputs;
    for (auto var_name : body_outputs_) {
      body_outputs << " " << var_name;
    }
    VLOG(6) << "body_outputs include: " << body_outputs.str();

    std::stringstream body_skip_gc_names;
    for (auto var_name : body_skip_gc_names_) {
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
      PADDLE_THROW(phi::errors::Unimplemented("unsupported type %d",
                                              inputs_[i]->Type()));
    }
  }
}

void WhileInstruction::CopyOutputsToBlockArgs() {
  for (size_t i = 0; i < body_block_->args_size(); ++i) {
    auto block_arg = body_block_->arg(i);
    auto var_name = body_inter_->GetNameByValue(block_arg);
    auto* inner_var = body_inter_->local_scope()->GetVar(var_name);

    if (outputs_[i]->IsType<phi::DenseTensor>()) {
      auto& src_tensor = outputs_[i]->Get<phi::DenseTensor>();
      auto* dst_tensor = inner_var->GetMutable<phi::DenseTensor>();
      dst_tensor->set_meta(src_tensor.meta());
      framework::TensorCopy(src_tensor, src_tensor.place(), dst_tensor);
    } else if (outputs_[i]->IsType<phi::TensorArray>()) {
      auto src_tensor_array = outputs_[i]->Get<phi::TensorArray>();
      auto* dst_tensor_array = inner_var->GetMutable<phi::TensorArray>();
      dst_tensor_array->set_type(src_tensor_array.dtype());
      dst_tensor_array->set_layout(src_tensor_array.layout());
      while (dst_tensor_array->size() < src_tensor_array.size()) {
        dst_tensor_array->emplace_back();
      }
      for (size_t id = 0; id < dst_tensor_array->size(); id++) {
        auto& src_tensor = src_tensor_array[id];
        phi::DenseTensor* tmp_dst_tensor = &dst_tensor_array->at(id);
        tmp_dst_tensor->set_meta(src_tensor.meta());
        framework::TensorCopy(src_tensor, src_tensor.place(), tmp_dst_tensor);
      }
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("unsupported type %d", inner_var->Type()));
    }
  }
  DeviceContext().Wait();
}

void WhileInstruction::ShareDatasToOutputs() {
  cond_var_->GetMutable<phi::DenseTensor>()->ShareDataWith(
      body_inter_->local_scope()
          ->GetVar(body_outputs_[0])
          ->Get<phi::DenseTensor>());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    auto& out_var_name = body_outputs_[i + 1];
    auto* out_var = body_inter_->local_scope()->GetVar(out_var_name);
    VLOG(6) << "share data from " << out_var_name << " -> " << i << " output";

    if (out_var->IsType<phi::DenseTensor>()) {
      outputs_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
          out_var->Get<phi::DenseTensor>());
    } else if (out_var->IsType<phi::TensorArray>()) {
      const auto& inner_array = out_var->Get<phi::TensorArray>();
      auto* output_array = outputs_[i]->GetMutable<phi::TensorArray>();
      *output_array = inner_array;
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("unsupported type %d", out_var->Type()));
    }

    VLOG(6) << "done";
  }
}

void WhileInstruction::Run() {
  ShareInputsToOutputs();
  VLOG(6) << "while instruction start loop ...";
  while (GetCondData(cond_var_->Get<phi::DenseTensor>())) {
    VLOG(6) << "while instruction pass args to body block";
    CopyOutputsToBlockArgs();
    VLOG(6) << "while instruction interpretercore run";
    body_inter_->Run({}, false);
    VLOG(6) << "while instruction get value form body block";
    ShareDatasToOutputs();
  }
  VLOG(6) << "while instruction run done";
}

}  // namespace framework
}  // namespace paddle
