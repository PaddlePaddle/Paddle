/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

// fake interfaces that has not be implemented by other modules.
struct OpRunContext {
  Scope* scope;
};

class OperatorBase {
 public:
  virtual ~OperatorBase() {}
  virtual void Run(OpRunContext* context) const = 0;
  virtual void InferShape(const Scope* scope) const = 0;

 protected:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
}

class RecurrentForwardOp {
 public:
  virtual void InferShape(const Scope* scope) = 0;
  /*
   * Forward run the RNN.
   *
   * NOTE the context's scope is not given until `Run` called, so step scopes'
   * father should be set/updated in this method.
   */
  virtual void Run(OpRunContext* contex) const = 0;

 protected:
  /*
   * Prepare inputs for each stepnet.
   */
  void ApplyInLinks(Scope* scope);

  /*
   * Process outputs of stepnets and merge to variables.
   */
  void ApplyOutLinks(Scope* scope);

  /*
   * Build a `Net` which is shared across all steps.
   */
  void BuildStepNet(Scope* scope);

  /*
   * Create a scope for each step, the context's scope is shared across all the
   * step scopes as the father scope. The step scopes will be stored in the
   * father scope as a variable.
   */
  void CreateScopes(Scope* scope);

  /*
   * Prepare steps' states and relations.
   */
  void PrepareStates(Scope* scope);

 protected:
  /*
   * these are defined in BaseOperator
   *
   * std::vector<std::string> inputs_;
   * std::vector<std::string> outputs_;
   */

  // State of a RNN (same as the role of `Momory` in PaddlePaddle)
  struct StateAttr {
    // name of current state variable
    std::string var;
    // name of previous step's state variable
    std::string pre_var;
    // name of the variable to init a state, which is store in context's scope.
    std::string boot_var;
  };
  std::vector<StateAttr> states_;
};

class RecurrentBackwardOp;
}  // namespace framework
}  // namespace paddle
