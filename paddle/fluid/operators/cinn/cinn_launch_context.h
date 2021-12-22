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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace operators {
namespace details {

using LoDTensor = framework::LoDTensor;
using CinnTensor = ::cinn::hlir::framework::Tensor;
using CinnScope = ::cinn::hlir::framework::Scope;

class CinnLaunchContext {
 public:
  explicit CinnLaunchContext(
      const std::unordered_map<std::string, std::string>& paddle2cinn_varmap,
      const std::shared_ptr<CinnScope>& cinn_scope);

  // explicitly update several environment variables captured
  // by callback of execution arguments
  void UpdateCapturedEnv(const framework::Scope& scope,
                         const platform::Place& place);

  // Return whether execution arguments has been initialized
  bool IsArgumentsInitialized() const;

  // Return whether a Paddle variable used on compiled kernels
  bool IsVariableUsed(const std::string& paddle_var_name) const;

  // Assign tensor buffer to input or output variables
  void AssignExternalVariable(const std::string& paddle_var_name);

  // Assign tensor buffer to internal variables
  void AssignInternalVariable(const std::string& cinn_var_name);

  // Extract internal variable names from CinnScope
  // by excluding used input and output variables
  std::unordered_set<std::string> GetInternalVariableNames();

  // Finalize all execution arguments and return them
  const std::map<std::string, cinn_pod_value_t>& FinalizeArguments() const;

 private:
  // Get CinnTensor with CINN variable name
  CinnTensor GetCinnTensor(const std::string& var_name);

  // Check whether tensors from Paddle and CINN of the same variable
  // are equivalent in type and dimension
  void CheckTensorEquivalent(const std::string& var_name,
                             const LoDTensor& paddle_tensor,
                             const CinnTensor& cinn_tensor);

  // Set an argument with (cinn name)->(cinn_buffer_t) pair
  void SetArgument(const std::string& cinn_var_name,
                   std::unique_ptr<cinn_buffer_t>&& buffer);

 private:
  const framework::Scope* cached_scope_ = nullptr;
  const platform::Place* cached_place_ = nullptr;
  std::unique_ptr<framework::Scope> cached_temp_scope_ = nullptr;

  // a variable name map from paddle to cinn
  const std::unordered_map<std::string, std::string>& paddle2cinn_varmap_;
  // the variable scope of cinn
  const std::shared_ptr<CinnScope> cinn_scope_;

  // all variables used by compiled executable program
  std::unordered_set<std::string> cinn_variable_names_;

  // because a cinn_pod_value_t does not own the cinn_buffer_t object,
  // an extra stroage is necessary to keep the object and it can
  // not be released until runtime program finish  execution.
  std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers_;

  // name to execution argument
  std::map<std::string, cinn_pod_value_t> name2argument_;
};

}  // namespace details
}  // namespace operators
}  // namespace paddle
