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
#include "paddle/pir/core/builtin_attribute.h"

namespace paddle {
namespace framework {

static DDim GetDimsDebug(const Scope& scope,
                         const std::string& name,
                         bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    return tensor.dims();
  } else if (var->IsType<phi::SelectedRows>()) {
    if (get_actual_dim) {
      return var->Get<phi::SelectedRows>().value().dims();
    } else {
      return var->Get<phi::SelectedRows>().GetCompleteDims();
    }
  } else if (var->IsType<Strings>()) {
    return DDim({static_cast<int64_t>(var->Get<Strings>().size())});
  } else {
    return DDim({-1});
  }
}

static bool VarInited(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) return false;
  return var->IsInitialized();
}

static std::string GetDtype(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return DataTypeToString(framework::TransToProtoVarType(tensor.dtype()));
  } else if (var->IsType<phi::SelectedRows>()) {
    auto tensor = var->Get<phi::SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return DataTypeToString(framework::TransToProtoVarType(tensor.dtype()));
    }
  } else if (var->IsType<Strings>()) {
    return "strings";
  } else {
    return "";
  }
}

static std::string GetPlace(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }
  auto to_string = [](const platform::Place& p) {
    std::stringstream sstream;
    sstream << p;
    return sstream.str();
  };

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return to_string(tensor.place());
  } else if (var->IsType<phi::SelectedRows>()) {
    auto tensor = var->Get<phi::SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return to_string(tensor.place());
    }
  } else {
    return "";
  }
}

static int GetRowSize(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return -1;
  }

  if (var->IsType<phi::SelectedRows>()) {
    return static_cast<int>(var->Get<phi::SelectedRows>().rows().size());
  }

  return -1;
}

static LoD GetLoDDebug(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    return tensor.lod();
  } else {
    return default_lod;
  }
}

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
    const std::unordered_map<pir::Value, std::vector<int>>& inputs) {
  input_index_ = inputs;
}

void InstructionBase::SetOutputs(
    const std::unordered_map<pir::Value, std::vector<int>>& outputs) {
  output_index_ = outputs;
}

void InstructionBase::InitInputsOutputsIds(
    ::pir::Operation* op,
    Scope* inner_scope,
    const std::unordered_map<pir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  std::unordered_map<pir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < op->num_operands(); i++) {
    pir::Value value = op->operand_source(i);
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
  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
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

std::string InstructionBase::DebugStringEx(
    const paddle::framework::Scope* scope,
    const std::unordered_map<::pir::Value, std::string>& value_2_var_name)
    const {
  std::stringstream ss;
  ss << "Op(" << Name() << "), inputs:{";

  const std::unordered_set<::pir::Value> no_need_buffer_vars = NoNeedBuffer();

  for (auto it = Inputs().begin(); it != Inputs().end();) {
    auto& input = *it;
    bool is_no_need_buffer_var = (!no_need_buffer_vars.empty() &&
                                  no_need_buffer_vars.count(input.first) > 0);
    auto var_name = value_2_var_name.at(input.first);
    ss << var_name;
    if (scope) {
      if (!VarInited(*scope, var_name)) {
        ss << "[uninited]";
      } else {
        int row_size = GetRowSize(*scope, var_name);
        if (row_size >= 0) {
          ss << "[row_size=" << row_size << "]";
        }
        std::string dtype = is_no_need_buffer_var ? "unknown_dtype"
                                                  : GetDtype(*scope, var_name);
        std::string place = is_no_need_buffer_var ? "unknown_place"
                                                  : GetPlace(*scope, var_name);
        ss << ":" << dtype;
        ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
        ss << "(" << GetLoDDebug(*scope, var_name) << ")";
        ss << "(" << place << ")";
      }
    }
    ++it;
    if (it != Inputs().end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = Outputs().begin(); it != Outputs().end();) {
    auto& output = *it;
    auto var_name = value_2_var_name.at(output.first);
    ss << var_name;
    if (scope) {
      if (!VarInited(*scope, var_name)) {
        ss << "[uninited]";
      } else {
        int row_size = GetRowSize(*scope, var_name);
        if (row_size >= 0) {
          ss << "[row_size=" << row_size << "]";
        }
        std::string dtype = GetDtype(*scope, var_name);
        ss << ":" << dtype;
        ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
        ss << "(" << GetLoDDebug(*scope, var_name) << ")";
        ss << "(" << GetPlace(*scope, var_name) << ")";
      }
    }
    ++it;
    if (it != Outputs().end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}
}  // namespace framework
}  // namespace paddle
