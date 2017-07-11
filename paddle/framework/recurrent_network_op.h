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
#include "paddle/framework/ddim.h"
#include "paddle/framework/enforce.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/variable.h"

// Remove when including operator.h
#include "paddle/framework/attr_checker.h"
#include "paddle/framework/op_desc.pb.h"

namespace paddle {
namespace framework {

// --------------------------------------------------------------------
// fake interfaces that has not be implemented by other modules.
// TODO keep updating according to other modules' designs.
typedef std::shared_ptr<Scope> ScopePtr;
struct OpRunContext {
  ScopePtr scope;
};

class OperatorBase {
 public:
  virtual ~OperatorBase() {}
  void Init(const OpDesc& op_desc, AttributeMap& attrs) {}
  virtual void Run(OpRunContext* context) const = 0;
  virtual void InferShape(ScopePtr scope) const = 0;
  inline Variable* Input(ScopePtr scope, int index) const {
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

// TODO replace this with Net's proto.
struct NetDesc {
  std::string name_;
  std::vector<OpDesc> op_descs;
};

class PlainNet {
 public:
  PlainNet() {}
  PlainNet(const NetDesc& desc) {
    for (const OpDesc& proto : desc.op_descs) {
      AddOp(proto);
    }
  }
  // PlainNet(const std::string desc) {}
  void AddOp(const OpDesc& desc);
  void Run(ScopePtr scope) {
    OpRunContext ctx;
    ctx.scope = scope;
    for (auto& op : ops_) {
      op->Run(&ctx);
    }
  }

 private:
  std::vector<std::unique_ptr<OperatorBase>> ops_;
};

// fake interfaces end
// --------------------------------------------------------------------
// The sequence format in RecurrentOp is Tensor<seq_len, batch_size, dim> now.
// TODO:
// 1. No-padding computing for sequences with indifinite length in one batch.
// 2. Hierarchical RNN for sequence with sub-sequence.
// 3. Multi-inputs with indifinate length for RecurrentOp.
// 4. More Complex RNN architecture, such as Gated Feedback RNN.
//    Refer to: https://arxiv.org/pdf/1502.02367.pdf
class RecurrentOp : public OperatorBase {
 public:
  /*
   * Initialize the recurrent operator from the operator protobuf
   * and attributes.
   */
  void Init(const OpDesc& op_desc, AttributeMap& attrs);

  virtual void InferShape(ScopePtr scope) const override {}

  /*
   * Forward run the RNN.
   *
   * NOTE the context's scope is not given until `Run` called, so step scopes'
   * father should be set/updated in this method.
   */
  virtual void Run(OpRunContext* contex) const override;

  virtual ~RecurrentOp() {}

 protected:
  /*
   * Prepare inputs for each stepnet.
   */
  void SegmentInputs(ScopePtr scope) const;

  /*
   * Process outputs of stepnets and merge to variables.
   */
  void ConcateOutputs(ScopePtr scope) const {};

  /*
   * Create a `Net` which is shared across all steps.
   */
  void CreateStepNet(ScopePtr scope) const;

  /*
   * Create a scope for each step, the context's scope is shared across all
   * the step scopes as the father scope. The step scopes will be stored in
   * the father scope as a variable whose name is specified by
   * `step_scopes_name_`.
   *
   * NOTE the scopes are reused by both the `Forward` and `Backward`, so just
   * create once and expand its size if more steps need.
   */
  void CreateScopes(ScopePtr scope) const;

  /*
   * Create memories in each step scope.
   */
  // void CreateMemories(ScopePtr scope) const;

  /*
   * Link memory in previous step scope to current scope.
   */
  void LinkMemories(ScopePtr scope, std::vector<ScopePtr>& step_scopes,
                    size_t step) const;

 private:
  /*
   * Memory of a RNN (same as the role of `Momory` in PaddlePaddle).
   *
   * Memory attributes cached by this op, dims will be infered from
   * boot memories in father scope. Other attributes are copied from Op's proto
   * attributes.
   */
  struct MemoryAttr {
    // name of current state variable
    std::string var;
    // name of previous step's state variable
    std::string pre_var;
    // name of the variables to init this memory (same role of `boot_layer` in
    // PaddlePaddle), which is store in father's scope.
    std::string boot_var;
    // this dim will infered from boot memories's tensor in the first step.
    DDim dims;
  };

  /*
   * The attributes in protobuf about the memory description and the initial
   * memory description are as follows. The number of initial memories should
   * equal to the memories number.
   *
   *   arg {
   *       name: “memories”
   *       strings: "hidden”
   *       strings: "state”
   *   }
   *   arg {
   *       name: “boot_memories”
   *       strings: "boot_hidden”
   *       strings: "boot_state”
   *   }
   */
  // TODO copy from OpBase's
  mutable std::vector<MemoryAttr> memory_attrs_;

  // this op's name, used as a unique key in father scope.
  // TODO repace it with OpBase's interface if supported.
  std::string name_;
  // name of rnn op's step net, the step net will be shared by both `Forward`
  // and `Backward`, so we store it as a variable in father's scope, with a
  // unique key specified by `net_name_`.
  std::string net_name_;
  // name of steps' scopes which is stored in father scope with a unique key
  // specified by `step_scopes_name_`.
  std::string step_scopes_name_;

  NetDesc step_net_desc_;
};

class RecurrentGradientOp;

}  // namespace framework
}  // namespace paddle
