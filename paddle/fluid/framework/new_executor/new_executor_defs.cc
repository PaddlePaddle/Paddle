// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {

VariableScope::VariableScope(Scope* scope) {
  // for @EMPTY@ variable
  name2id_[kEmptyVarName] = kEmptyVarIndex;
  var_list_.push_back(nullptr);
  vec_meta_info_.emplace_back(0, nullptr);
  scope_ = scope;
  PADDLE_ENFORCE_NE(
      scope,
      nullptr,
      platform::errors::PreconditionNotMet(
          "You have passed a nullptr to construct VariableScope."));
}

VariableScope::~VariableScope() = default;

Scope* VariableScope::GetMutableScope() const { return scope_; }

Scope* VariableScope::GetMutableLocalScope() const { return local_scope_; }

void VariableScope::SetScope(Scope* scope) { scope_ = scope; }

void VariableScope::SetLocalScope(Scope* local_scope) {
  VLOG(4) << "Set local scope: " << local_scope;
  local_scope_ = local_scope;
}

// Get variable id by name, return -1 if not found
int VariableScope::GetIdByName(const std::string& name) const {
  auto it = name2id_.find(name);
  if (it != name2id_.end()) {
    return it->second;
  }
  return -1;
}

// Get variable name by id, return "" if not found
std::string VariableScope::GetNameById(int id) const {
  // NOTE(zhiqiu): do not use vec_meta_info_[id].vardesc_->Name() since
  // vec_meta_info_[id] may be nullptr,
  // typically when the target variable is not existed in the original program
  // desc, but created by interpretercore.
  // For example, created and used by d2h_copy or h2d_copy operator.
  auto it = std::find_if(name2id_.begin(),
                         name2id_.end(),
                         [id](const auto& pair) { return pair.second == id; });
  if (it != name2id_.end()) {
    return it->first;
  }
  return "";
}

bool VariableScope::HasVar(const std::string& name) const {
  return name2id_.find(name) != name2id_.end();
}

int VariableScope::VarId(const std::string& name) const {
  CheckExist(name);
  return name2id_.at(name);
}

Variable* VariableScope::VarRef(int id) const { return var_list_[id]; }

size_t VariableScope::VarSize() const { return name2id_.size(); }

void VariableScope::AddVar(const std::string& name,
                           framework::VarDesc* var_desc) {
  if (!HasVar(name)) {
    auto id = VarSize();
    name2id_[name] = static_cast<int>(id);
    vec_meta_info_.emplace_back(0, var_desc);
    if (local_scope_ != nullptr) {
      var_list_.push_back(local_scope_->FindVar(name));
    } else {
      var_list_.push_back(scope_->FindVar(name));
    }
    PADDLE_ENFORCE_EQ(
        var_list_.size(),
        name2id_.size(),
        platform::errors::InvalidArgument(
            "The size of var_list and name2id map should be equal"));
  }
}

void VariableScope::SetVarDesc(const std::string& name,
                               framework::VarDesc* var_desc) {
  CheckExist(name);
  vec_meta_info_[VarId(name)].var_desc_ = var_desc;
}

paddle::framework::VarDesc* VariableScope::VarDesc(
    const std::string& name) const {
  return VarDesc(VarId(name));
}

paddle::framework::VarDesc* VariableScope::VarDesc(int id) const {
  CheckExist(id);
  return vec_meta_info_[id].var_desc_;
}

void VariableScope::SetVarSikpInplace(const std::string& name, bool skip) {
  CheckExist(name);
  vec_meta_info_[VarId(name)].sikp_inplace_ = skip;
}

bool VariableScope::GetVarSikpInplace(int id) const {
  CheckExist(id);
  return vec_meta_info_[id].sikp_inplace_;
}

void VariableScope::CheckExist(int id) const {
  PADDLE_ENFORCE_LT(id,
                    name2id_.size(),
                    platform::errors::PreconditionNotMet(
                        "Required var_id < %d, but received var_id = %d.",
                        name2id_.size(),
                        id));
}

void VariableScope::CheckExist(const std::string& name) const {
  PADDLE_ENFORCE_EQ(
      HasVar(name),
      true,
      platform::errors::NotFound("%s not in VariableScope.", name));
}

Instruction::Instruction(size_t id,
                         OpFuncNode&& op_func_node,
                         const platform::DeviceContext& dev_ctx)
    : is_artificial_(false),
      id_(id),
      op_func_node_(op_func_node),
      dev_ctx_(dev_ctx) {
  if (op_func_node.operator_base_ != nullptr &&
      op_func_node.operator_base_->Type() == "depend") {
    is_artificial_ = true;
  }

  if (op_func_node_.phi_kernel_ != nullptr) {
    pre_define_context_ = true;
  }
  PADDLE_ENFORCE_GE(id,
                    0,
                    platform::errors::PreconditionNotMet(
                        "Required id >= 0, but received id = %d", id));
}

void Instruction::WaitEvent(const Place& place) const {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(place)) {
    return;
  }

  VLOG(6) << "Deal StreamWaitEventOrSync for " << this->OpBase()->Type();

  for (const EventInter& event_iter : events_to_wait_) {
    platform::RecordEvent record(
        "WaitStreamEvent", platform::TracerEventType::UserDefined, 10);
    VLOG(6) << "Wait instruction: " << event_iter.instr_id_
            << " 's event with waiter_type: " << event_iter.waiter_type_;
    event_iter.event_->Wait(event_iter.waiter_type_, &dev_ctx_);
  }
}

void Instruction::RecordEvent(const Place& place) const {
  platform::RecordEvent record(
      "RecordStreamEvent", platform::TracerEventType::UserDefined, 10);
  if (event_to_record_) {
    VLOG(6) << "Record event at instruction: " << id_;
    event_to_record_->event_->Record(&dev_ctx_);
  }
}

const std::map<std::string, std::vector<int>>& Instruction::Inputs() const {
  return op_func_node_.input_index;
}

const std::map<std::string, std::vector<int>>& Instruction::Outputs() const {
  return op_func_node_.output_index;
}

OpKernelComputeFunc Instruction::KernelFunc() const {
  return op_func_node_.kernel_func_;
}

phi::Kernel* Instruction::PhiKernel() const {
  return op_func_node_.phi_kernel_;
}

OpFuncType Instruction::KernelType() const { return op_func_node_.type_; }

const std::map<int, int>& Instruction::InplaceBackMap() const {
  return op_func_node_.inplace_back_map;
}

OperatorBase* Instruction::OpBase() const {
  auto op_base = op_func_node_.operator_base_;
  PADDLE_ENFORCE_NOT_NULL(
      op_base,
      platform::errors::PreconditionNotMet("op_base shall not be nullptr."));
  return op_base.get();
}

bool Instruction::OpBaseValid() const {
  return op_func_node_.operator_base_ != nullptr;
}

void Instruction::AddGCCheckVar(size_t id) { gc_check_vars_.push_back(id); }

const std::vector<size_t>& Instruction::GCCheckVars() const {
  return gc_check_vars_;
}

void Instruction::ResetContext(const VariableValueMap& in_vars,
                               const VariableValueMap& out_vars) {
  runtime_ctx_.reset(new RuntimeContext(in_vars, out_vars));
  infershape_ctx_.reset(
      new RuntimeInferShapeContext(*OpBase(), *runtime_ctx_.get()));
  // NOTE: Because execution_ctx_ is constructed by `scope&`, so we fake an
  // empty here to avoid illegal local reference.
  static framework::Scope scope_;
  execution_ctx_.reset(
      new ExecutionContext(*OpBase(), scope_, dev_ctx_, *runtime_ctx_.get()));
}

void Instruction::ResetContextWithScope(const VariableValueMap& in_vars,
                                        const VariableValueMap& out_vars,
                                        const framework::Scope& scope) {
  runtime_ctx_.reset(new RuntimeContext(in_vars, out_vars));
  infershape_ctx_.reset(
      new RuntimeInferShapeContext(*OpBase(), *runtime_ctx_.get()));
  execution_ctx_.reset(
      new ExecutionContext(*OpBase(), scope, dev_ctx_, *runtime_ctx_.get()));
}

std::shared_ptr<RuntimeContext> Instruction::InnerRuntimeContext() const {
  return runtime_ctx_;
}

std::shared_ptr<RuntimeInferShapeContext> Instruction::InnerInferShapeContext()
    const {
  return infershape_ctx_;
}

std::shared_ptr<ExecutionContext> Instruction::InnerExecutionContext() const {
  return execution_ctx_;
}

const platform::DeviceContext& Instruction::DeviceContext() const {
  return dev_ctx_;
}

const std::vector<std::pair<Variable*, Variable*>>& Instruction::InplaceInfo()
    const {
  return vec_inplace_in_to_out_;
}

void Instruction::AddInplace(Variable* in, Variable* out) {
  vec_inplace_in_to_out_.emplace_back(in, out);
}

void Instruction::ClearInplace() { vec_inplace_in_to_out_.clear(); }

}  // namespace framework
}  // namespace paddle
