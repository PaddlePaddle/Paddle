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

#include <google/protobuf/text_format.h>
#include "paddle/framework/attr_checker.h"
#include "paddle/framework/enforce.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/variable.h"

namespace paddle {
namespace framework {

// --------------------------------------------------------------------
// fake interfaces that has not be implemented by other modules.
// TODO keep updating according to other modules' designs.
struct OpRunContext {
  Scope* scope;
};

// TODO replace this with Net's proto.
struct NetDesc {
  std::string name_;
};

class PlainNet {
 public:
  PlainNet() {}
  PlainNet(const NetDesc& desc) {}
  PlainNet(const std::string desc) {}
  void Run(Scope* scope) {}
};

class OperatorBase {
 public:
  virtual ~OperatorBase() {}
  virtual void Run(OpRunContext* context) const = 0;
  virtual void InferShape(const Scope* scope) const = 0;
  inline Variable* Input(Scope* scope, int index) const {
    return scope->GetVariable(inputs_[index]);
  };

  template <typename T>
  inline const T GetAttr(const std::string& name) const {
    return boost::get<T>(attrs_.at(name));
  }

 protected:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
};
// fake interfaces end
// --------------------------------------------------------------------

class RecurrentOp : public OperatorBase {
 public:
  RecurrentOp(NetDesc& net_desc)
      : name_(net_desc.name_),
        net_name_(net_desc.name_ + "__net__"),
        step_scopes_name_(net_desc.name_ + "__step_scopes_") {}

  virtual void InferShape(const Scope* scope) const override;

  /*
   * Forward run the RNN.
   *
   * NOTE the context's scope is not given until `Run` called, so step scopes'
   * father should be set/updated in this method.
   */
  virtual void Run(OpRunContext* contex) const override;

 protected:
  /*
   * Prepare inputs for each stepnet.
   */
  void SegmentInputs(Scope* scope) const;

  /*
   * Process outputs of stepnets and merge to variables.
   */
  void ConcateOutputs(Scope* scope) const;

  /*
   * Create a `Net` which is shared across all steps.
   */
  void CreateStepNet(Scope* scope) const;

  /*
   * Create a scope for each step, the context's scope is shared across all
   * the step scopes as the father scope. The step scopes will be stored in
   * the father scope as a variable whose name is specified by
   * `step_scopes_name_`.
   *
   * NOTE the scopes are reused by both the `Forward` and `Backward`, so just
   * create once and expand its size if more steps need.
   */
  void CreateScopes(Scope* scope) const;

  /*
   * Create memories in each step scope.
   */
  void CreateMemories(Scope* scope) const;

  /*
   * Link memory in previous step scope to current scope.
   */
  // void LinkMemories(Scope* scope) const;

 private:
  /*
   * these are defined in BaseOperator
   *
   * std::vector<std::string> inputs_;
   * std::vector<std::string> outputs_;
   */

  // Memory of a RNN (same as the role of `Momory` in PaddlePaddle)
  struct MemoryAttr {
    // name of current state variable
    std::string var;
    // name of previous step's state variable
    std::string pre_var;
    // name of the variables to init this memory (same role of `boot_layer` in
    // PaddlePaddle), which is store in father's scope.
    std::string boot_var;
  };

  std::vector<MemoryAttr> memory_attrs_;

  // this op's name, used as a unique key in father scope.
  // TODO repace it with OpBase's interface if supported.
  std::string name_;
  // name of rnn op's step net, the step net will be shared by both `Forward`
  // and `Backward`, so we store it as a variable in father's scope, with a
  // unique key specified by `net_name_`.
  const std::string net_name_;
  // name of steps' scopes which is stored in father scope with a unique key
  // specified by `step_scopes_name_`.
  const std::string step_scopes_name_;

  const NetDesc step_net_desc_;
};

class RecurrentGradientOp;

}  // namespace framework
}  // namespace paddle
