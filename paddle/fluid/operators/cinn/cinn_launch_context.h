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
namespace operators::details {

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

  // Return whether a Paddle variable used in cinn execution
  bool IsVariableUsed(const std::string& var_name) const;

  // Assign tensor buffer to input or output variables
  void AssignExternalVariable(const std::string& var_name);

  // Assign tensor buffer to internal variables
  void AssignInternalVariable(const std::string& var_name);

  // Extract internal variable names from CinnScope
  // by excluding used input and output variables
  std::unordered_set<std::string> GetInternalVariableNames();

  // Finalize all execution arguments and return them
  const std::map<std::string, cinn_pod_value_t>& FinalizeArguments() const;

  cinn_buffer_t* GetCinnBufferOfVar(const std::string& var_name);

 private:
  // Get CinnTensor with CINN argument name
  CinnTensor GetCinnTensor(const std::string& arg_name);
  // Build the name maps of paddle->cinn and cinn->paddle
  // in reverse for all variables used in cinn execution
  void BuildVarNameMap(
      const std::unordered_map<std::string, std::string>& compiled_varmap,
      const std::unordered_set<std::string>& argument_names);

  // Check whether tensors from Paddle and CINN of the same variable
  // are equivalent in type and dimension
  void CheckTensorEquivalent(const std::string& var_name,
                             const LoDTensor& paddle_tensor,
                             const CinnTensor& cinn_tensor);

  // Set an argument with (cinn name)->(cinn_buffer_t) pair
  void SetArgument(const std::string& arg_name,
                   std::unique_ptr<cinn_buffer_t>&& buffer);

 private:
  const framework::Scope* cached_scope_ = nullptr;
  const platform::Place* cached_place_ = nullptr;
  std::unique_ptr<framework::Scope> cached_temp_scope_ = nullptr;

  // Generally speaking, in this class, a variable is refer to paddle
  // Variable while an cinn variable is called argument.
  // a name map from paddle variable to cinn execution argument
  std::unordered_map<std::string, std::string> paddle2cinn_varmap_;
  // a name map from cinn execution argument to paddle variable
  std::unordered_map<std::string, std::string> cinn2paddle_varmap_;
  // all names of cinn arguments used in compiled executable program
  std::unordered_set<std::string> cinn_argument_names_;
  // the variable scope of cinn
  const std::shared_ptr<CinnScope> cinn_scope_;

  // because a cinn_pod_value_t does not own the cinn_buffer_t object,
  // an extra stroage is necessary to keep the object and it can
  // not be released until the runtime program finish execution.
  std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers_;

  // this map saves all execution arguments with their cinn names as key,
  // and it is passed to the Execute interface of a cinn runtime program.
  std::map<std::string, cinn_pod_value_t> name2argument_;
};

}  // namespace operators::details
}  // namespace paddle
