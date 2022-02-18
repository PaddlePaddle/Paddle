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

#include "paddle/fluid/operators/cinn/cinn_launch_context.h"
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/pten/core/ddim.h"

namespace paddle {
namespace operators::details {

using LoDTensor = framework::LoDTensor;

CinnLaunchContext::CinnLaunchContext(
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap,
    const std::shared_ptr<CinnScope>& cinn_scope)
    : cinn_scope_(cinn_scope) {
  // generate all names of the cinn execution arguments
  auto var_names = cinn_scope_->var_names();
  cinn_argument_names_.reserve(var_names.size());
  std::transform(
      var_names.begin(), var_names.end(),
      std::inserter(cinn_argument_names_, cinn_argument_names_.end()),
      [](const auto& name_view) { return std::string(name_view.data()); });
  // build name map between the original variables and compiled ones
  BuildVarNameMap(paddle2cinn_varmap, cinn_argument_names_);
}

void CinnLaunchContext::BuildVarNameMap(
    const std::unordered_map<std::string, std::string>& compiled_varmap,
    const std::unordered_set<std::string>& argument_names) {
  for (const auto& x : compiled_varmap) {
    if (!argument_names.count(x.second)) {
      // exclude variables not used
      continue;
    }
    // copy to local paddle2cinn map
    paddle2cinn_varmap_.emplace(x.first, x.second);
    // add an entry to local cinn2paddle map reversely
    auto res = cinn2paddle_varmap_.emplace(x.second, x.first);
    PADDLE_ENFORCE_EQ(
        res.second, true,
        platform::errors::InvalidArgument(
            "Cinn variable(%s) maps to more than one paddle variable(%s,%s)",
            x.second, res.first->second, x.first));
  }
  // supplement the relations of the remain variables
  // not appearing in above map, which are internal variables
  // and here we use the names from cinn compiled.
  for (const auto& var_name : argument_names) {
    if (!cinn2paddle_varmap_.count(var_name)) {
      cinn2paddle_varmap_.emplace(var_name, var_name);
      paddle2cinn_varmap_.emplace(var_name, var_name);
    }
  }

  PADDLE_ENFORCE_EQ(
      paddle2cinn_varmap_.size(), cinn2paddle_varmap_.size(),
      platform::errors::PreconditionNotMet(
          "Size of variables is not euqal, paddle[%ld] vs cinn[%ld]",
          paddle2cinn_varmap_.size(), cinn2paddle_varmap_.size()));
}

void CinnLaunchContext::UpdateCapturedEnv(const framework::Scope& scope,
                                          const platform::Place& place) {
  if (std::addressof(scope) == cached_scope_ &&
      std::addressof(place) == cached_place_) {
    VLOG(4) << "Captured scope:" << cached_scope_ << ", place:" << cached_place_
            << " are not changed";
    return;
  }
  cached_scope_ = std::addressof(scope);
  cached_place_ = std::addressof(place);
  cached_temp_scope_ = scope.NewTmpScope();
  VLOG(4) << "Captured env is update, scope:" << cached_scope_ << "->"
          << std::addressof(scope) << ", place:" << cached_place_ << "->"
          << std::addressof(place);
}

bool CinnLaunchContext::IsArgumentsInitialized() const {
  if (hold_buffers_.empty() || name2argument_.empty()) {
    return false;
  }
  return true;
}

bool CinnLaunchContext::IsVariableUsed(const std::string& var_name) const {
  return paddle2cinn_varmap_.count(var_name) > 0;
}

CinnTensor CinnLaunchContext::GetCinnTensor(const std::string& arg_name) {
  PADDLE_ENFORCE_GT(cinn_argument_names_.count(arg_name), 0,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not found in cinn scope.", arg_name));
  return cinn_scope_->GetTensor(arg_name);
}

std::unordered_set<std::string> CinnLaunchContext::ExtractInternalVarNames(
    const std::vector<std::string>& input_var_names,
    const std::vector<std::string>& output_var_names) {
  std::unordered_set<std::string> remain_var_names;
  remain_var_names.reserve(paddle2cinn_varmap_.size());
  std::transform(paddle2cinn_varmap_.begin(), paddle2cinn_varmap_.end(),
                 std::inserter(remain_var_names, remain_var_names.end()),
                 [](const auto& name_pair) { return name_pair.first; });

  // exclude the input variables and output variables
  auto exclude_names_fn = [&remain_var_names](const std::string& var_name) {
    remain_var_names.erase(var_name);
  };
  std::for_each(input_var_names.begin(), input_var_names.end(),
                exclude_names_fn);
  std::for_each(output_var_names.begin(), output_var_names.end(),
                exclude_names_fn);
  return remain_var_names;
}

void CinnLaunchContext::CheckTensorEquivalent(const std::string& var_name,
                                              const LoDTensor& paddle_tensor,
                                              const CinnTensor& cinn_tensor) {
  // check dimension
  auto cinn_dims = pten::make_ddim(cinn_tensor->shape().data());
  PADDLE_ENFORCE_EQ(paddle_tensor.dims(), cinn_dims,
                    platform::errors::PreconditionNotMet(
                        "Tensors' shape in variable(%s) are not equivalent, "
                        "paddle's shape = [%s], but cinn's shape = [%s].",
                        var_name, paddle_tensor.dims(), cinn_dims));

  // TODO(CtfGo): check the underlying data type after CINN ready
}

void CinnLaunchContext::AssignExternalVariable(const std::string& var_name) {
  PADDLE_ENFORCE_EQ(IsVariableUsed(var_name), true,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not applied in cinn", var_name));
  const auto& cinn_arg_name = paddle2cinn_varmap_.at(var_name);

  const auto& paddle_tensor = cached_scope_->GetVar(var_name)->Get<LoDTensor>();
  CinnTensor cinn_tensor = GetCinnTensor(cinn_arg_name);
  if (paddle_tensor.IsInitialized()) {
    CheckTensorEquivalent(var_name, paddle_tensor, cinn_tensor);
  }

  auto cinn_buffer = std::make_unique<cinn_buffer_t>();
  // assign dimensions and alloc/free callback of cinn_buffer_t
  cinn_buffer->resize(cinn_tensor->shape().data().data(),
                      cinn_tensor->shape().data().size());
  cinn_buffer->external_malloc = new std::function<int(void*, cinn_buffer_t*)>(
      [this, var_name](void* ctx, cinn_buffer_t* buffer) {
        auto* tensor = cached_scope_->GetVar(var_name)->GetMutable<LoDTensor>();
        tensor->Resize(framework::DDim(buffer->dims, buffer->dimensions));
        buffer->memory = reinterpret_cast<uint8_t*>(
            tensor->mutable_data<float>(*cached_place_));
        return 0;
      });

  // external variables will be recycled by global gc, so do nothing here
  cinn_buffer->external_free = new std::function<int(void*, cinn_buffer_t*)>(
      [](void* ctx, cinn_buffer_t* buffer) {
        // Do nothing
        return 0;
      });

  return AppendArgument(cinn_arg_name, std::move(cinn_buffer));
}

void CinnLaunchContext::AssignInternalVariable(const std::string& var_name) {
  PADDLE_ENFORCE_EQ(IsVariableUsed(var_name), true,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not applied in cinn", var_name));
  const auto& cinn_arg_name = paddle2cinn_varmap_.at(var_name);

  CinnTensor cinn_tensor = GetCinnTensor(cinn_arg_name);
  auto cinn_buffer = std::make_unique<cinn_buffer_t>();
  // assign dimensions and alloc/free callback of cinn_buffer_t
  cinn_buffer->resize(cinn_tensor->shape().data().data(),
                      cinn_tensor->shape().data().size());

  cinn_buffer->external_malloc = new std::function<int(void*, cinn_buffer_t*)>(
      [this, var_name](void* ctx, cinn_buffer_t* buffer) {
        auto* tensor =
            cached_temp_scope_->Var(var_name)->GetMutable<LoDTensor>();
        tensor->Resize(framework::DDim(buffer->dims, buffer->dimensions));
        buffer->memory = reinterpret_cast<uint8_t*>(
            tensor->mutable_data<float>(*cached_place_));
        return 0;
      });

  // internal variables should release its buffer immediately
  // if no instruction use it
  cinn_buffer->external_free = new std::function<int(void*, cinn_buffer_t*)>(
      [this, var_name](void* ctx, cinn_buffer_t* buffer) {
        auto* tensor =
            cached_temp_scope_->GetVar(var_name)->GetMutable<LoDTensor>();
        tensor->clear();
        return 0;
      });
  return AppendArgument(cinn_arg_name, std::move(cinn_buffer));
}

void CinnLaunchContext::AppendArgument(
    const std::string& arg_name, std::unique_ptr<cinn_buffer_t>&& buffer) {
  name2argument_.emplace(arg_name, buffer.get());
  hold_buffers_.emplace_back(std::move(buffer));
  VLOG(4) << string::Sprintf(
      "Append an argument:name(%s),dims(%s),argument size:(%lu)", arg_name,
      framework::DDim(buffer->dims, buffer->dimensions).to_str(),
      name2argument_.size());
}

const std::map<std::string, cinn_pod_value_t>&
CinnLaunchContext::FinalizeArguments() const {
  // Check all execution parameters are assigned valued.
  std::for_each(cinn_argument_names_.begin(), cinn_argument_names_.end(),
                [this](const auto& arg_name) {
                  PADDLE_ENFORCE_GT(
                      name2argument_.count(arg_name), 0,
                      platform::errors::NotFound(
                          "Argument(%s) is missed for execution", arg_name));
                });
  return name2argument_;
}

cinn_buffer_t* CinnLaunchContext::GetCinnBufferOfVar(
    const std::string& var_name) {
  auto it = paddle2cinn_varmap_.find(var_name);
  PADDLE_ENFORCE_NE(
      it, paddle2cinn_varmap_.end(),
      platform::errors::InvalidArgument(
          "Variable(%s) not found in compilation result", var_name));
  auto res = name2argument_.find(it->second);
  PADDLE_ENFORCE_NE(res, name2argument_.end(),
                    platform::errors::NotFound(
                        "Argument(%s) not be initialized", it->second));
  return static_cast<cinn_buffer_t*>(res->second);
}

}  // namespace operators::details
}  // namespace paddle
