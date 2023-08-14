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

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/ir/core/builtin_attribute.h"

namespace paddle {
namespace framework {

InstructionBase::InstructionBase(size_t id, const platform::Place& place) {
  id_ = id;

  is_artificial_ = false;

  if (platform::is_cpu_place(place)) {
    type_ = OpFuncType::kCpuSync;
  } else {
    PADDLE_ENFORCE_EQ(
        interpreter::IsSupportedHeterPlace(place),
        true,
        phi::errors::Fatal("Unsupported current place %s", place));
    type_ = OpFuncType::kGpuAsync;
  }

  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place);
}

OpFuncType InstructionBase::KernelType() const { return type_; }

const platform::DeviceContext& InstructionBase::DeviceContext() const {
  return *dev_ctx_;
}

void InstructionBase::RecordEvent(const Place& place) const {
  platform::RecordEvent record(
      "RecordStreamEvent", platform::TracerEventType::UserDefined, 10);
  if (event_to_record_) {
    VLOG(6) << "Record event at instruction: " << id_;
    event_to_record_->event_->Record(dev_ctx_);
  }
}

void InstructionBase::WaitEvent(const Place& place) const {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(place)) {
    return;
  }
  for (const EventInter& event_iter : events_to_wait_) {
    platform::RecordEvent record(
        "WaitStreamEvent", platform::TracerEventType::UserDefined, 10);
    VLOG(6) << "Wait instruction: " << event_iter.instr_id_
            << " 's event with waiter_type: " << event_iter.waiter_type_;
    event_iter.event_->Wait(event_iter.waiter_type_, dev_ctx_);
  }
}

void InstructionBase::AddGCCheckVar(size_t id) { gc_check_vars_.push_back(id); }

const std::vector<size_t>& InstructionBase::GCCheckVars() const {
  return gc_check_vars_;
}

const std::vector<std::pair<Variable*, Variable*>>&
InstructionBase::InplaceInfo() const {
  return vec_inplace_in_to_out_;
}

void InstructionBase::AddInplace(Variable* in, Variable* out) {
  vec_inplace_in_to_out_.emplace_back(in, out);
}

void InstructionBase::ClearInplace() { vec_inplace_in_to_out_.clear(); }

void InstructionBase::SetInputs(
    const std::unordered_map<ir::Value, std::vector<int>>& inputs) {
  input_index_ = inputs;
}

void InstructionBase::SetOutputs(
    const std::unordered_map<ir::Value, std::vector<int>>& outputs) {
  output_index_ = outputs;
}

void InstructionBase::InitInputsOutputsIds(
    ::ir::Operation* op,
    Scope* inner_scope,
    const std::unordered_map<::ir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<::ir::StrAttribute>().AsString();
  std::unordered_map<ir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < op->num_operands(); i++) {
    ir::Value value = op->operand_source(i);
    if (value) {
      PADDLE_ENFORCE_NE(
          value_2_var_name.find(value),
          value_2_var_name.end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              op_name));
      std::vector<int> inputs_id = GetValueIds(value,
                                               inner_scope,
                                               value_2_var_name,
                                               var_name_2_id,
                                               variable_2_var_name);
      inputs.emplace(value, inputs_id);
    }
  }
  SetInputs(inputs);
  VLOG(8) << "finish process inputs_index";
  std::unordered_map<ir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    ir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_NE(
          value_2_var_name.find(value),
          value_2_var_name.end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              op_name));
      std::vector<int> outputs_id = GetValueIds(value,
                                                inner_scope,
                                                value_2_var_name,
                                                var_name_2_id,
                                                variable_2_var_name);
      outputs.emplace(value, outputs_id);
    }
  }
  SetOutputs(outputs);
  VLOG(8) << "finish process outputs_index";
}

}  // namespace framework
}  // namespace paddle
