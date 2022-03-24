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
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/ddim.h"

// type declaration forward
struct cinn_buffer_t;
struct cinn_pod_value_t;
namespace cinn::hlir::framework {
class Tensor;
class Scope;
class Program;
}  // namespace cinn::hlir::framework

namespace paddle {
namespace framework {
class ProgramDesc;
class Scope;
class VarDesc;

namespace ir {
class Graph;
}  // namespace ir

namespace paddle2cinn {
class CinnCompiledObject;
}  // namespace paddle2cinn
}  // namespace framework

namespace operators::details {

using CinnTensor = ::cinn::hlir::framework::Tensor;
using CinnScope = ::cinn::hlir::framework::Scope;
using CinnCompiledObject = framework::paddle2cinn::CinnCompiledObject;

// This class is used to cache some reusable data among repeated
// executions for efficiency and it also provides easy interfaces
// to get details of the compilation result.
// A object of this class is constructed and saved in the
// compilation cache once a graph compiled by CINN.
// Generally speaking, here, a variable is refer to a Paddle
// Variable while a CINN variable is called an Argument.
class CinnLaunchContext {
 public:
  explicit CinnLaunchContext(const framework::ir::Graph& graph,
                             const CinnCompiledObject& compiled_obj);

  // Initialize a ParallelExecutor to execute the runtime graph,
  // it will be constructed in the first call, and just update
  // the execution scope in the following usage.
  framework::ParallelExecutor* InitializePE(const platform::Place& place,
                                            framework::Scope* scope);

  // explicitly update several environment variables captured
  // by callback of execution arguments
  void UpdateCapturedEnv(const framework::Scope& scope,
                         const platform::Place& place);

  // Return whether a Paddle variable used in cinn execution
  bool IsVariableUsed(const std::string& var_name) const;

  // Check the equiality in type and dimension between the tensor
  // in Paddle and the compiled tensor returned by CINN of a same variable
  void CheckTensorEquivalent(const std::string& var_name,
                             const framework::LoDTensor& paddle_tensor);

  // Return the name list of variables skipped eager deletion
  const std::vector<std::string>& GetSkipEagerVars() const {
    return skip_eager_vars_;
  }

  // Return internal variable names list
  const std::unordered_set<std::string>& GetInternalVarNames() const {
    return internal_var_names_;
  }

  // Finalize all execution arguments and return the name->argument map
  const std::map<std::string, cinn_pod_value_t>& FinalizeArguments() const {
    return name2argument_;
  }

  // Return the cinn_buffer_t* of a specific variable
  cinn_buffer_t* GetCinnBufferOfVar(const std::string& var_name);

 private:
  // Get corresponding compiled tensor of a Paddle variable name
  CinnTensor GetCinnTensorOfVar(const std::string& var_name);

  // Build the name maps of paddle->cinn and cinn->paddle
  // in reverse for all variables used in cinn execution
  void BuildVarNameMap(
      const std::unordered_map<std::string, std::string>& compiled_varmap,
      const std::unordered_set<std::string>& argument_names);

  // Extract internal variable names from all applied variables
  // in execution by excluding the input and output variables
  std::unordered_set<std::string> ExtractInternalVarNames(
      const std::vector<std::string>& input_var_names,
      const std::vector<std::string>& output_var_names);

  // Initialize each execution argument with a cinn_buffer_t
  void InitializeArguments();

  // Assign tensor buffer to input or output variables
  void AssignExternalVariable(const std::string& var_name);

  // Assign tensor buffer to internal variables
  void AssignInternalVariable(const std::string& var_name);

  // Construct a Paddle ProgramDesc with the CINN runtime
  // instructions included in the compiled CINN Program
  framework::ProgramDesc BuildCompiledProgram(
      const framework::ir::Graph& graph,
      const CinnCompiledObject& compiled_obj);

 private:
  const framework::Scope* cached_scope_ = nullptr;
  const platform::Place* cached_place_ = nullptr;
  std::unique_ptr<framework::Scope> cached_temp_scope_ = nullptr;

  // a name map from paddle variables to cinn execution arguments
  std::unordered_map<std::string, std::string> paddle2cinn_varmap_;
  // a name map from cinn execution arguments to paddle variables
  std::unordered_map<std::string, std::string> cinn2paddle_varmap_;
  // a list of internal variable names in Paddle
  std::unordered_set<std::string> internal_var_names_;
  // the names of the cinn arguments used in compiled executable program
  std::unordered_set<std::string> cinn_argument_names_;
  // TODO(CtfGo): remove this list after fixing batch_norm bug
  // due to duplicate association in the same variable.
  std::vector<std::string> initialized_beforehand_vars_;
  // the variable scope compiled from cinn
  const std::shared_ptr<CinnScope> cinn_scope_;

  // the ir::Graph object converted from the program compiled by CINN
  std::unique_ptr<framework::ir::Graph> runtime_graph_;
  // a ParallelExecutor to execute the runtime graph
  std::unique_ptr<framework::ParallelExecutor> parallel_executor_;
  // the name list of skip_eager_vars in runtime
  std::vector<std::string> skip_eager_vars_;

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
