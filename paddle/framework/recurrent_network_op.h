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

#include <glog/logging.h>
#include "paddle/framework/op_desc.pb.h"

namespace paddle {
namespace framework {

// --------------------------------------------------------------------
// fake interfaces that has not be implemented by other modules.
// TODO keep updating according to other modules' designs.
typedef std::shared_ptr<Scope> ScopePtr;
struct OpContext {
  ScopePtr scope;
};

class OperatorBase {
 public:
  virtual ~OperatorBase() {}
  void Init(const OpDesc& op_desc, AttributeMap& attrs) { attrs_ = attrs; }
  virtual void Run(OpContext* context) const = 0;
  virtual void InferShape(ScopePtr scope) const = 0;
  inline Variable* Input(ScopePtr scope, std::string name) const {
    return scope->GetVariable(name);
  };

  template <typename T>
  inline const T& GetAttr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }

 protected:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
};

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
    OpContext ctx;
    ctx.scope = scope;
    for (auto& op : ops_) {
      op->Run(&ctx);
    }
  }

 private:
  std::vector<std::unique_ptr<OperatorBase>> ops_;
};

namespace details {

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
};

};  // namespace details

// fake interfaces end
// --------------------------------------------------------------------
// The sequence format in RecurrentOp is Tensor<seq_len, batch_size, dim> now.
// TODO:
// 1. No-padding computing for sequences with indifinite length in one batch.
// 2. Hierarchical RNN for sequence with sub-sequence.
// 3. External Memory.
// 4. More Complex RNN architecture, such as Gated Feedback RNN.
//    Refer to: https://arxiv.org/pdf/1502.02367.pdf

/*
 * RecurrentOp inputs stored in proto:
 * - in_links : real inputs that need to be segmented to steps.
 * - boot memories
 * - all weights in step net
 * - step net
 *
 * outputs:
 * - out_links : real outputs
 * - step scopes
 *
 * Attributes stored in AttributeMap:
 * - in_links: vector<int>
 * - boot_memories: vector<int>
 * - step_net: int
 * - in_link_alias: vector<string>  the alias of in_links in step net.
 * - out_link_alias: vector<string> the alias of out_links in step net
 * - memories: vector<string> the memory names
 * - pre_memories: vector<string> the previous memory names
 *
 * see RecurrentOpProtoAndCheckerMaker
 */

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
  virtual void Run(OpContext* contex) const override;

  virtual ~RecurrentOp() {}

 protected:
  /*
   * Get the max sequence length of the scope.
   */
  size_t GetMaxSeqLen(ScopePtr scope) const;

  /*
   * Prepare inputs for each stepnet.
   */
  void SegmentInputs(ScopePtr scope) const;

  /*
   * Process outputs of stepnets and merge to variables.
   */
  void ConcatOutputs(ScopePtr scope) const;

  /*
   * the step scopes as the father scope. The step scopes will be stored in
   * the father scope as a variable whose name is specified by
   * `step_scopes_name_`.
   *
   * NOTE the scopes are reused by both the `Forward` and `Backward`, so just
   * create once and expand its size if more steps need.
   */
  void CreateScopes(ScopePtr scope) const;

  /*
   * Get the step scopes.
   */
  inline const std::vector<ScopePtr>& GetStepScopes(ScopePtr scope) const {
    return *(scope->GetVariable(step_scopes_name_))
                ->GetMutable<std::vector<ScopePtr>>();
  }

  /*
   * Link memory in previous step scope to current scope.
   */
  void LinkMemories(std::vector<ScopePtr>& step_scopes, size_t step_id) const;

 private:
  /*
   * The attributes in protobuf about the memory description and the initial
   * memory description are as follows. The number of initial memories should
   * equal to the memories number.
   *
   *   arg {
   *       name: "memories"
   *       strings: "hidden"
   *       strings: "state"
   *   }
   *   arg {
   *       name: “pre_memories"
   *       strings: "pre_hidden"
   *       strings: "pre_state"
   *   }
   *   arg {
   *       name: “boot_memories"
   *       strings: "boot_hidden"
   *       strings: "boot_state"
   *   }
   */
  mutable std::vector<details::MemoryAttr> memory_attrs_;

  // name of rnn op's step net, the step net will be shared by both `Forward`
  // and `Backward`, so we store it as a variable in father's scope, with a
  // unique key specified by `net_name_`.
  std::string net_name_;
  // name of steps' scopes which is stored in father scope with a unique key
  // specified by `step_scopes_name_`.
  std::string step_scopes_name_;
  // real inputs that need to be segmented.
  std::vector<std::string> inlinks_;
  std::vector<std::string> outlinks_;

  std::vector<std::string> in_link_alias_;
  std::vector<std::string> out_link_alias_;
};

/*
 * RNN's backward alogorithm.
 *
 * To accelerate the development of RecurrentBackwardOp, we decouple RNN's
 * algorithm and `RecurrentBackwardAlgorithm`, the former contains the core
 * implementation of a RNN, and will keep stable even if the framework changes a
 * lot, and the latter is a wrapper acts like an dapter for it to make RNN an
 * operator.
 */
class RecurrentBackwardAlgorithm {
 public:
 private:
  // stepnet for backward
  // NOTE this stepnet is created by others and should insert AddOp for its
  // weights gradient updating, RNN backward just run it.
  std::string stepnet_name_;
  // step scopes that shared by both the forward and backward operators.
  std::string step_scopes_name_;

  // inputs(gradients of forward operator's outputs) that need to be segmented
  // for each step.
  std::vector<std::string> inlinks_;
  // outputs(gradients of forward operator's inputs) of each step that need to
  // be concated.
  std::vector<std::string> outlinks_;

  // alias to avoid duplicate keys in scopes.
  std::vector<std::string> inlink_alias_;
  std::vector<std::string> outlink_alias_;

  // NOTE the first step's boot memories' gradients should be outputed.
  std::vector<details::MemoryAttr> memories_;
};

}  // namespace framework
}  // namespace paddle
