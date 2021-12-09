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
#include <functional>
#include <vector>

namespace paddle {
namespace operators {
namespace details {

CinnLaunchContext::CinnLaunchContext(
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap,
    const std::shared_ptr<CinnScope>& cinn_scope)
    : paddle2cinn_varmap_(paddle2cinn_varmap), cinn_scope_(cinn_scope) {
  auto var_names = cinn_scope_->var_names();
  cinn_variable_names_.reserve(var_names.size());
  std::transform(
      var_names.begin(), var_names.end(),
      std::inserter(cinn_variable_names_, cinn_variable_names_.end()),
      [](const auto& name_view) { return std::string(name_view.data()); });
}

bool CinnLaunchContext::IsVariableUsed(const std::string& paddle_name) {
  return paddle2cinn_varmap_.count(paddle_name) > 0 &&
         cinn_variable_names_.count(paddle2cinn_varmap_.at(paddle_name)) > 0;
}

CinnTensor CinnLaunchContext::GetCinnTensor(const std::string& var_name) {
  PADDLE_ENFORCE_GT(cinn_variable_names_.count(var_name), 0,
                    platform::errors::NotFound(
                        "Variable(%s) not found in cinn scope.", var_name));
  return cinn_scope_->GetTensor(var_name);
}

std::unordered_set<std::string> CinnLaunchContext::GetInternalVariableNames() {
  std::unordered_set<std::string> all_parameters(cinn_variable_names_);
  std::for_each(name2argument_.begin(), name2argument_.end(),
                [&all_parameters](const auto& name2arg) {
                  all_parameters.erase(name2arg.first);
                });
  return all_parameters;
}

void CinnLaunchContext::CheckTensorEquivalent(const std::string& paddle_name,
                                              const LoDTensor& paddle_tensor,
                                              const CinnTensor& cinn_tensor) {
  // check dimension
  auto cinn_dims = framework::make_ddim(cinn_tensor->shape().data());
  PADDLE_ENFORCE_EQ(paddle_tensor.dims(), cinn_dims,
                    platform::errors::PreconditionNotMet(
                        "Tensors' shape in variable(%s) are not equivalent, "
                        "paddle's shape = [%s], but cinn's shape = [%s].",
                        paddle_name, paddle_tensor.dims(), cinn_dims));

  // TODO(CtfGo): check the underlying data type after CINN ready
}

void CinnLaunchContext::AssignExternalVariable(const std::string& paddle_name,
                                               const platform::Place& place,
                                               LoDTensor* paddle_tensor) {
  PADDLE_ENFORCE_EQ(IsVariableUsed(paddle_name), true,
                    platform::errors::InvalidArgument(
                        "Paddle variable(%s) not used by cinn", paddle_name));

  const auto& cinn_name = paddle2cinn_varmap_.at(paddle_name);
  CinnTensor cinn_tensor = GetCinnTensor(cinn_name);
  if (!paddle_tensor->IsInitialized()) {
    paddle_tensor->Resize(framework::make_ddim(cinn_tensor->shape().data()));
  }
  CheckTensorEquivalent(paddle_name, *paddle_tensor, cinn_tensor);
  return SetArgument(cinn_name, place, /* free_mem_callback = */ false,
                     paddle_tensor);
}

void CinnLaunchContext::AssignInternalVariable(const std::string& cinn_name,
                                               const platform::Place& place,
                                               LoDTensor* paddle_tensor) {
  PADDLE_ENFORCE_GT(cinn_variable_names_.count(cinn_name), 0,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not found in cinn socpe.", cinn_name));
  CinnTensor cinn_tensor = GetCinnTensor(cinn_name);
  if (!paddle_tensor->IsInitialized()) {
    paddle_tensor->Resize(framework::make_ddim(cinn_tensor->shape().data()));
  }
  CheckTensorEquivalent(cinn_name, *paddle_tensor, cinn_tensor);
  return SetArgument(cinn_name, place, /* free_mem_callback = */ true,
                     paddle_tensor);
}

std::unique_ptr<cinn_buffer_t> CinnLaunchContext::ShareTensorWithCinnBuffer(
    const platform::Place& place, bool free_mem_callback, LoDTensor* tensor) {
  // convert paddle dimensions array to cinn format
  std::vector<cinn_dimension_t> cinn_dims(tensor->dims().size());
  for (auto i = 0; i < tensor->dims().size(); ++i) {
    cinn_dims[i] = static_cast<cinn_dimension_t>(tensor->dims().at(i));
  }

  auto cinn_buffer = std::make_unique<cinn_buffer_t>();
  // assign size and memory
  cinn_buffer->resize(cinn_dims.data(), cinn_dims.size());

  cinn_buffer->external_malloc = new std::function<int(void*, cinn_buffer_t*)>(
      [place, tensor](void* ctx, cinn_buffer_t* buffer) {
        buffer->memory =
            reinterpret_cast<uint8_t*>(tensor->mutable_data<float>(place));
        return 0;
      });

  if (free_mem_callback) {
    cinn_buffer->external_free = new std::function<int(void*, cinn_buffer_t*)>(
        [tensor](void* ctx, cinn_buffer_t* buffer) {
          tensor->clear();
          return 0;
        });
    return cinn_buffer;
  }

  cinn_buffer->external_free = new std::function<int(void*, cinn_buffer_t*)>(
      [](void* ctx, cinn_buffer_t* buffer) {
        // Do nothing
        return 0;
      });
  return cinn_buffer;
}

void CinnLaunchContext::SetArgument(const std::string& cinn_name,
                                    const platform::Place& place,
                                    bool free_mem_callback,
                                    LoDTensor* paddle_tensor) {
  auto buffer =
      ShareTensorWithCinnBuffer(place, free_mem_callback, paddle_tensor);
  name2argument_.emplace(cinn_name, buffer.get());
  hold_buffers_.emplace_back(std::move(buffer));
  VLOG(4) << "SetArgument-" << name2argument_.size() << ": "
          << "name(" << cinn_name << "), dims(" << paddle_tensor->dims()
          << ").";
}

const std::map<std::string, cinn_pod_value_t>&
CinnLaunchContext::FinalizeArguments() const {
  // Check all execution parameters are assigned valued.
  std::for_each(cinn_variable_names_.begin(), cinn_variable_names_.end(),
                [this](const auto& var_name) {
                  PADDLE_ENFORCE_GT(name2argument_.count(var_name), 0,
                                    platform::errors::InvalidArgument(
                                        "Variable(%s) is missed for launching "
                                        "compiled program execution",
                                        var_name));
                });
  return name2argument_;
}

}  // namespace details
}  // namespace operators
}  // namespace paddle
