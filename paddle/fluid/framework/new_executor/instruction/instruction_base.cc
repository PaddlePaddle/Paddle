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
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace paddle::framework {

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
  auto to_string = [](const phi::Place& p) {
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
  } else if (var->IsType<VariableRefArray>()) {
    return var->Get<VariableRefArray>().size();
  } else if (var->IsType<phi::TensorArray>()) {
    return var->Get<phi::TensorArray>().size();
  }

  return -1;
}

static LegacyLoD GetLoDDebug(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LegacyLoD({{}});

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

static double GetDenseTensorEleSum(const Scope& scope,
                                   const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (var->IsType<phi::DenseTensor>() &&
      var->Get<phi::DenseTensor>().initialized()) {
    phi::DenseTensor cpu_tensor;
    phi::CPUPlace place;
    paddle::framework::TensorCopy(
        var->Get<phi::DenseTensor>(), place, &cpu_tensor);
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(var->Get<phi::DenseTensor>().place());
    dev_ctx.Wait();
    double sum = 0.0;
    for (int64_t i = 0; i < cpu_tensor.numel(); i++) {
      if (cpu_tensor.dtype() == phi::DataType::FLOAT32) {
        sum += static_cast<double>(cpu_tensor.data<float>()[i]);
      } else if (cpu_tensor.dtype() == phi::DataType::FLOAT64) {
        sum += static_cast<double>(cpu_tensor.data<double>()[i]);
      } else if (cpu_tensor.dtype() == phi::DataType::INT32) {
        sum += static_cast<double>(cpu_tensor.data<int32_t>()[i]);
      } else if (cpu_tensor.dtype() == phi::DataType::INT64) {
        sum += static_cast<double>(cpu_tensor.data<int64_t>()[i]);
      } else if (cpu_tensor.dtype() == phi::DataType::FLOAT16) {
        const phi::dtype::float16* data =
            cpu_tensor.data<phi::dtype::float16>();
        sum += static_cast<double>(data[0]);
      } else if (cpu_tensor.dtype() == phi::DataType::BOOL) {
        sum += static_cast<double>(cpu_tensor.data<bool>()[i]);
      } else {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }
    return sum;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

InstructionBase::InstructionBase(size_t id, const phi::Place& place)
    : next_instrs_in_different_thread_(),
      next_instrs_in_same_thread_(),
      events_to_wait_info_(),
      events_to_wait_(),
      gc_check_vars_(),
      eager_gc_vars_(),
      vec_inplace_in_to_out_(),
      inplace_back_map_(),
      input_index_(),
      output_index_(),
      no_need_buffer_values_() {
  id_ = id;

  if (phi::is_cpu_place(place)) {
    type_ = OpFuncType::kCpuSync;
  } else {
    PADDLE_ENFORCE_EQ(
        interpreter::IsSupportedHeterPlace(place),
        true,
        common::errors::Fatal("Unsupported current place %s", place));
    type_ = OpFuncType::kGpuAsync;
  }

  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place);
}

OpFuncType InstructionBase::KernelType() const { return type_; }

const phi::DeviceContext& InstructionBase::DeviceContext() const {
  return *dev_ctx_;
}

void InstructionBase::RecordEvent(const Place& place) const {
  if (event_to_record_) {
    phi::RecordEvent record(
        "RecordStreamEvent", phi::TracerEventType::UserDefined, 10);
    VLOG(6) << "Record event at instruction: " << id_;
    event_to_record_->event_->Record(dev_ctx_);
  }
}

void InstructionBase::WaitEvent(const Place& place) const {
  for (const EventInter& event_iter : events_to_wait_) {
    // If InterpreterCore in on CPUPlace, do nothing.
    if (phi::is_cpu_place(place)) {
      continue;
    }
    phi::RecordEvent record(
        "WaitStreamEvent", phi::TracerEventType::UserDefined, 10);
    VLOG(6) << "Wait instruction: " << event_iter.instr_id_
            << " 's event with waiter_type: " << event_iter.waiter_type_;
    event_iter.event_->Wait(event_iter.waiter_type_, dev_ctx_);
  }
}

void InstructionBase::AddGCCheckVar(size_t id) { gc_check_vars_.push_back(id); }

const std::vector<size_t>& InstructionBase::GCCheckVars() const {
  return gc_check_vars_;
}

void InstructionBase::AddEagerGCVar(Variable* var) {
  if (var->IsType<VariableRefArray>()) {
    auto array = var->Get<VariableRefArray>();
    for (size_t i = 0; i < array.size(); ++i) {
      AddEagerGCVar(const_cast<Variable*>(array.at(i)));
    }
  } else {
    if (std::find(eager_gc_vars_.begin(), eager_gc_vars_.end(), var) ==
        eager_gc_vars_.end()) {
      eager_gc_vars_.push_back(var);
    }
  }
}

const std::vector<Variable*>& InstructionBase::EagerGCVars() const {
  // NOTE(chenxi67): eager_gc_vars_ contains the vars that need to be gc. Some
  // vars in Instruction Node are created temporarily and are not the input or
  // output of an OP (e.g. copy_var created by TuplePushOp). We cannot determine
  // whether they need to be gc by analyzing OP(using GCCheckVars() function).
  // These vars are added to eager_gc_vars_ and directly gc.
  return eager_gc_vars_;
}

void InstructionBase::ClearEagerGCVars() { eager_gc_vars_.clear(); }

const std::vector<std::pair<const Variable*, Variable*>>&
InstructionBase::InplaceInfo() const {
  return vec_inplace_in_to_out_;
}

void InstructionBase::AddInplace(const Variable* in, Variable* out) {
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
    ::pir::Operation* op, const ValueExecutionInfo& value_exec_info) {
  auto op_attributes = op->attributes();
  std::string op_name;
  if (op_attributes.count("op_name")) {
    op_name =
        op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }
  std::unordered_map<pir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < op->num_operands(); i++) {
    pir::Value value = op->operand_source(i);
    if (value) {
      PADDLE_ENFORCE_EQ(
          value_exec_info.HasValue(value),
          true,
          common::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              op_name));
      std::vector<int> inputs_id = GetValueIds(value, value_exec_info);
      inputs.emplace(value, inputs_id);
    }
  }
  SetInputs(inputs);
  VLOG(8) << "finish process inputs_index";
  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_EQ(
          value_exec_info.HasValue(value),
          true,
          common::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              op_name));
      std::vector<int> outputs_id = GetValueIds(value, value_exec_info);
      outputs.emplace(value, outputs_id);
    }
  }

  const auto value_2_var_name_map = value_exec_info.GetValue2VarName();
  for (auto inplace_var_pair : this->InplaceInfo()) {
    for (auto item : value_2_var_name_map) {
      if (item.second == value_exec_info.GetVarName(inplace_var_pair.first)) {
        std::vector<int> outputs_id = GetValueIds(item.first, value_exec_info);
        outputs.emplace(item.first, outputs_id);
        break;
      }
    }
  }
  SetOutputs(outputs);
  VLOG(8) << "finish process outputs_index";
}

std::string InstructionBase::DebugStringEx(
    const paddle::framework::Scope* scope,
    ValueExecutionInfo* value_exe_info) const {
  std::stringstream ss;
  ss << "Op(" << Name() << "), inputs:{";

  const std::unordered_set<::pir::Value> no_need_buffer_vars = NoNeedBuffer();

  for (auto it = Inputs().begin(); it != Inputs().end();) {
    auto& input = *it;
    bool is_no_need_buffer_var = (!no_need_buffer_vars.empty() &&
                                  no_need_buffer_vars.count(input.first) > 0);
    auto var_name = value_exe_info->GetVarName(input.first);
    ss << var_name << ":[";
    if (scope) {
      if (!VarInited(*scope, var_name)) {
        ss << "uninited";
      } else {
        std::string dtype = is_no_need_buffer_var ? "unknown_dtype"
                                                  : GetDtype(*scope, var_name);
        std::string place = is_no_need_buffer_var ? "unknown_place"
                                                  : GetPlace(*scope, var_name);
        ss << "dtype=" << dtype << ";";
        ss << "place=" << place << ";";

        ss << "dim=" << GetDimsDebug(*scope, var_name, true) << ";";
        ss << "lod=" << GetLoDDebug(*scope, var_name) << ";";
        int row_size = GetRowSize(*scope, var_name);
        if (row_size >= 0) {
          ss << "row_size=" << row_size << ";";
        }
      }
    }
    ++it;
    if (it != Inputs().end()) {
      ss << "], ";
    } else {
      ss << "]";
    }
  }
  ss << "}, outputs:{";
  for (auto it = Outputs().begin(); it != Outputs().end();) {
    auto& output = *it;
    auto var_name = value_exe_info->GetVarName(output.first);
    ss << var_name << ":[";
    if (scope) {
      if (!VarInited(*scope, var_name)) {
        ss << "uninited";
      } else {
        std::string dtype = GetDtype(*scope, var_name);
        ss << "dtype=" << dtype << ";";
        ss << "place=" << GetPlace(*scope, var_name) << ";";
        ss << "dim=" << GetDimsDebug(*scope, var_name, true) << ";";
        ss << "lod=" << GetLoDDebug(*scope, var_name) << ";";
        int row_size = GetRowSize(*scope, var_name);
        if (row_size >= 0) {
          ss << "row_size=" << row_size << ";";
        }
      }
    }
    ++it;
    if (it != Outputs().end()) {
      ss << ", ";
    } else {
      ss << "]";
    }
  }
  ss << "}.";
  return ss.str();
}
}  // namespace paddle::framework
