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
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"

// type declaration forward
struct cinn_buffer_t;
struct cinn_pod_value_t;
namespace cinn::hlir::framework {
class Tensor;
class Scope;
class Program;
}  // namespace cinn::hlir::framework

namespace paddle {
namespace operators::details {

using CinnTensor = ::cinn::hlir::framework::Tensor;
using CinnScope = ::cinn::hlir::framework::Scope;

// This class is used to cache some reusable data among repeated
// executions for efficiency and it also provides easy interfaces
// to get details of the compilation result.
// A object of this class is constructed and saved in the
// compilation cache once a graph compiled by CINN.
// Generally speaking, here, a variable is refer to a Paddle
// Variable while a CINN variable is called an Argument.
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

  // Extract internal variable names from all applied variables
  // in execution by excluding the input and output variables
  std::unordered_set<std::string> ExtractInternalVarNames();

  // Finalize all execution arguments and return the name->argument map
  const std::map<std::string, cinn_pod_value_t>& FinalizeArguments() const;

  // Return the cinn_buffer_t* of a specific variable
  cinn_buffer_t* GetCinnBufferOfVar(const std::string& var_name);

 private:
  // Get CinnTensor with CINN argument name
  CinnTensor GetCinnTensor(const std::string& arg_name);
  // Build the name maps of paddle->cinn and cinn->paddle
  // in reverse for all variables used in cinn execution
  void BuildVarNameMap(
      const std::unordered_map<std::string, std::string>& compiled_varmap,
      const std::unordered_set<std::string>& argument_names);

  // Check whether the tensor in Paddle and the compiled
  // tensor returned by CINN of a same variable
  // are equivalent in type and dimension
  void CheckTensorEquivalent(const std::string& var_name,
                             const framework::LoDTensor& paddle_tensor,
                             const CinnTensor& cinn_tensor);

  // Append an argument with (cinn name)->(cinn_buffer_t) pair
  void AppendArgument(const std::string& arg_name,
                      std::unique_ptr<cinn_buffer_t>&& buffer);

 private:
  const framework::Scope* cached_scope_ = nullptr;
  const platform::Place* cached_place_ = nullptr;
  std::unique_ptr<framework::Scope> cached_temp_scope_ = nullptr;

  // a name map from paddle variables to cinn execution arguments
  std::unordered_map<std::string, std::string> paddle2cinn_varmap_;
  // a name map from cinn execution arguments to paddle variables
  std::unordered_map<std::string, std::string> cinn2paddle_varmap_;
  // the names of the cinn arguments used in compiled executable program
  std::unordered_set<std::string> cinn_argument_names_;
  // the variable scope compiled from cinn
  const std::shared_ptr<CinnScope> cinn_scope_;

  // because a cinn_pod_value_t does not own a cinn_buffer_t object,
  // an extra stroage is necessary to keep those objects and they can
  // not be released until the runtime program finish execution.
  std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers_;

  // this map saves all execution arguments with their cinn names as key,
  // and it is passed to the Execute interface of a cinn runtime program.
  std::map<std::string, cinn_pod_value_t> name2argument_;
};

}  // namespace operators::details
}  // namespace paddle
