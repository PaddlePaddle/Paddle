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

#include "paddle/fluid/operators/cinn_launch_op_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace details {

Name2ConstTensor GetConstTensors(
    const Scope& scope, const std::vector<std::string>& variable_names) {
  Name2ConstTensor name2tensor;
  for (const auto& var_name : variable_names) {
    auto* var_ptr = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var_ptr, platform::errors::NotFound("Variable(%s) not found in Scope.",
                                            var_name));
    PADDLE_ENFORCE_EQ(var_ptr->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "Variable(%s) is not LoDTensor that is "
                          "the only supported by compiler now.",
                          var_name));
    name2tensor.emplace(var_name, &var_ptr->Get<framework::LoDTensor>());
  }

  return name2tensor;
}

Name2CinnTensor GetCompiledTensors(
    const std::vector<std::string>& paddle_var_names,
    const CinnScope& compiled_scope,
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap) {
  Name2CinnTensor name2tensor;
  for (const auto& pd_name : paddle_var_names) {
    PADDLE_ENFORCE_GT(paddle2cinn_varmap.count(pd_name), 0,
                      platform::errors::NotFound(
                          "the corresponding compiled one of variable(%s) "
                          "not found in compilation result.",
                          pd_name));
    const auto& cinn_name = paddle2cinn_varmap.at(pd_name);
    PADDLE_ENFORCE_NOT_NULL(
        compiled_scope.FindVar(cinn_name),
        platform::errors::NotFound("Variable(%s) not found in compiled scope.",
                                   pd_name));
    name2tensor.emplace(pd_name, compiled_scope.GetTensor(cinn_name));
  }
  return name2tensor;
}

void CheckTensorEquivalent(const Name2ConstTensor& paddle_tensors,
                           const Name2CinnTensor& compiled_tensors) {
  for (const auto& name2tensor : paddle_tensors) {
    const auto& pd_name = name2tensor.first;
    const auto* paddle_tensor = name2tensor.second;
    PADDLE_ENFORCE_EQ(
        paddle_tensor->IsInitialized(), true,
        platform::errors::InvalidArgument(
            "The tensor in variable(%s) is not initialized.", pd_name));

    PADDLE_ENFORCE_GT(compiled_tensors.count(pd_name), 0,
                      platform::errors::NotFound(
                          "the corresponding compiled tensor of variable(%s) "
                          "not found in compilation result.",
                          pd_name));
    const auto& cinn_tensor = compiled_tensors.at(pd_name);
    auto compiled_dim = framework::make_ddim(cinn_tensor->shape().data());

    PADDLE_ENFORCE_EQ(paddle_tensor->dims(), compiled_dim,
                      platform::errors::InvalidArgument(
                          "The tensor dimension in variable(%s) "
                          "is not equivalent, paddle is [%s] "
                          "but compiled result is [%s].",
                          pd_name, paddle_tensor->dims(), compiled_dim));
    // TODO(CtfGo): check the underlying data type is equivalent
  }
}

void InitializeOutputVar(const Scope& scope, const platform::Place& place,
                         const Name2CinnTensor& compiled_tensors) {
  for (const auto& name2tensor : compiled_tensors) {
    const auto& pd_name = name2tensor.first;
    const auto& cinn_tensor = name2tensor.second;
    auto* var_ptr = scope.FindVar(pd_name);
    PADDLE_ENFORCE_NOT_NULL(
        var_ptr, platform::errors::NotFound("Variable(%s) not found in scope.",
                                            pd_name));
    auto* paddle_tensor = var_ptr->GetMutable<LoDTensor>();
    if (!paddle_tensor->IsInitialized()) {
      paddle_tensor->Resize(framework::make_ddim(cinn_tensor->shape().data()));
      // TODO(CtfGo): support mutable corresponding c++ type
      //              with the compilation type
      paddle_tensor->mutable_data<float>(place);
      VLOG(2) << "Variable(%s) is initialized using compilation result";
    }
  }
}

std::vector<std::string> SeperateTempVar(
    const CinnScope& compiled_scope,
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap,
    const std::vector<std::string>& input_var_names,
    const std::vector<std::string>& output_var_names) {
  std::unordered_set<std::string> all_paddle_names, all_cinn_names;
  for_each(paddle2cinn_varmap.begin(), paddle2cinn_varmap.end(),
           [&all_paddle_names](const auto& name_pd2cinn) {
             all_paddle_names.insert(name_pd2cinn.first);
           });
  auto cinn_names_view = compiled_scope.var_names();
  for_each(cinn_names_view.begin(), cinn_names_view.end(),
           [&all_cinn_names](const auto& str_view) {
             all_cinn_names.emplace(str_view.data(), str_view.size());
           });

  auto exclude_fn = [&](const auto& pd_name) {
    PADDLE_ENFORCE_EQ(all_paddle_names.erase(pd_name), 1,
                      platform::errors::NotFound(
                          "The corresponding compiled one of variable(%s) "
                          "not found in compilation result.",
                          pd_name));
    PADDLE_ENFORCE_EQ(all_cinn_names.erase(paddle2cinn_varmap.at(pd_name)), 1,
                      platform::errors::NotFound(
                          "Variable(%s) not found in compiled scope", pd_name));
  };
  for_each(input_var_names.begin(), input_var_names.end(), exclude_fn);
  for_each(output_var_names.begin(), output_var_names.end(), exclude_fn);

  if (all_cinn_names.empty()) {
    VLOG(2) << "No temporary variable is needed during "
               "execution in cinn runtime program";
    return {};
  }

  return {all_cinn_names.begin(), all_cinn_names.end()};
}

void InitializeTempVar(const std::vector<std::string>& variable_names,
                       const CinnScope& compiled_scope,
                       const platform::Place& place, Scope* temp_scope) {
  for (const auto& var_name : variable_names) {
    PADDLE_ENFORCE_NOT_NULL(
        compiled_scope.FindVar(var_name),
        platform::errors::NotFound(
            "Temporary variable(%s) not found in compiled scope", var_name));

    const auto& cinn_tensor = compiled_scope.GetTensor(var_name);
    // use the same variable name defined by CINN
    auto* var_ptr = temp_scope->Var(var_name);
    auto* paddle_tensor = var_ptr->GetMutable<LoDTensor>();
    auto compiled_ddim = framework::make_ddim(cinn_tensor->shape().data());
    paddle_tensor->Resize(compiled_ddim);
    // TODO(CtfGo): support mutable corresponding c++ type with the compilation
    // type
    paddle_tensor->mutable_data<float>(place);
    VLOG(2) << "Add temporary variable(%s), dimension is [%s]." << var_name
            << compiled_ddim;
  }
}

void SharePaddleTensorWithCinnBuffer(LoDTensor* paddle_tensor,
                                     cinn_buffer_t* cinn_buffer) {
  cinn_buffer->resize(
      reinterpret_cast<const cinn_dimension_t*>(paddle_tensor->dims().Get()),
      paddle_tensor->dims().size());
  cinn_buffer->memory =
      reinterpret_cast<uint8_t*>(paddle_tensor->data<float>());
}

void AppendExecutionArguments(
    const Scope& scope, const std::vector<std::string>& variable_names,
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap,
    std::map<std::string, cinn_pod_value_t>* name2argument,
    std::vector<std::unique_ptr<cinn_buffer_t>>* hold_buffers) {
  for (const auto& pd_name : variable_names) {
    auto* var_ptr = scope.FindVar(pd_name);
    PADDLE_ENFORCE_NOT_NULL(
        var_ptr, platform::errors::NotFound("Variable(%s) not found in Scope.",
                                            pd_name));
    auto* paddle_tensor = var_ptr->GetMutable<LoDTensor>();
    // if not found a paddle variable in the map,
    // which means it is a temporary variable extra added,
    // so the paddle name is same with cinn
    const auto& cinn_name = paddle2cinn_varmap.count(pd_name)
                                ? paddle2cinn_varmap.at(pd_name)
                                : pd_name;

    std::unique_ptr<cinn_buffer_t> buffer_ptr(new cinn_buffer_t());
    SharePaddleTensorWithCinnBuffer(paddle_tensor, buffer_ptr.get());
    name2argument->emplace(cinn_name, buffer_ptr.get());
    hold_buffers->emplace_back(std::move(buffer_ptr));
  }
}

}  // namespace details
}  // namespace operators
}  // namespace paddle
