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

#include "paddle/operators/recurrent_op.h"

#include <glog/logging.h>
#include <cstring>
#include <sstream>

#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace operators {

void RecurrentAlgorithm::InferShape(const Scope& scope) const {
  seq_len_ = scope.FindVar((arg_->inlinks[0]).external)
                 ->GetMutable<Tensor>()
                 ->dims()[0];
  CreateScopes(scope);
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_,
                     true /*infer_shape_mode*/);
  InitMemories(step_scopes[0], true /*infer_shape_mode*/);
  Variable* net = scope.FindVar(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");
  for (size_t i = 0; i < seq_len_; i++) {
    if (i > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, i, -1,
                        true /*infer_shape_mode*/);
    }
    net->GetMutable<NetOp>()->InferShape(*step_scopes[i]);
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_,
                     true /*infer_shape_mode*/);
}

void RecurrentAlgorithm::Run(const Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_,
                     false /*infer_shape_mode*/);
  InitMemories(step_scopes[0], false /*infer_shape_mode*/);
  Variable* net = scope.FindVar(arg_->step_net);

  for (size_t step_id = 0; step_id < seq_len_; step_id++) {
    if (step_id > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, -1,
                        false /*infer_shape_mode*/);
    }
    net->GetMutable<NetOp>()->Run(*step_scopes[step_id], dev_ctx);
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_,
                     false /*infer_shape_mode*/);
}

void RecurrentAlgorithm::CreateScopes(const Scope& scope) const {
  // TODO(xxx) Only two scopes are needed for inference, this case will be
  // supported later.
  auto step_scopes =
      scope.FindVar(arg_->step_scopes)->GetMutable<std::vector<Scope*>>();

  if (seq_len_ > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len_; ++i) {
      auto& step_scope = scope.NewScope();

      // Now all variables in scope must be created outside of op.
      auto net_op = scope.FindVar(arg_->step_net)->GetMutable<NetOp>();
      for (auto& input : net_op->inputs_) {
        // the weight are located in parent scope
        if (!step_scope.FindVar(input)) step_scope.NewVar(input);
      }
      for (auto& output : net_op->outputs_) {
        step_scope.NewVar(output);
      }
      step_scopes->emplace_back(&step_scope);
    }
  }
}

void RecurrentAlgorithm::InitMemories(Scope* step_scope,
                                      bool infer_shape_mode) const {
  for (auto& attr : arg_->memories) {
    Tensor* pre_mem = step_scope->NewVar(attr.pre_var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(step_scope->FindVar(attr.boot_var) != nullptr,
                   "memory [%s]'s boot variable [%s] not exists", attr.var,
                   attr.boot_var);
    Tensor* boot_mem = step_scope->FindVar(attr.boot_var)->GetMutable<Tensor>();
    if (infer_shape_mode) {
      pre_mem->Resize(boot_mem->dims());
    } else {
      pre_mem->ShareDataWith<float>(*boot_mem);
    }
  }
}

const rnn::ArgumentName RecurrentOp::kArgName{
    "step_net", "step_scopes",  "inlinks",
    "outlinks", "inlink_alias", "outlink_alias",
    "memories", "pre_memories", "boot_memories"};

const rnn::ArgumentName RecurrentGradientOp::kArgName{
    "step_net",    "step_scopes",  "outlink@grad",
    "inlink@grad", "inlink_alias", "outlink_alias",
    "memories",    "pre_memories", "boot_memories@grad"};

void RecurrentOp::Init() {
  OperatorBase::Init();
  std::unique_ptr<rnn::Argument> arg(new rnn::Argument());
  rnn::InitArgument(kArgName, arg.get(), *this);
  alg_.Init(std::move(arg));
}

class RecurrentAlgorithmProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  RecurrentAlgorithmProtoAndCheckerMaker(OpProto* proto,
                                         OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    const auto& name = RecurrentOp::kArgName;
    // inputs and outputs stored in proto
    AddInput(name.inlinks,
             "the inputs that need to be segmented for each step.")
        .SetMultiple();
    AddInput(name.boot_memories, "variables to initialize memories.")
        .SetMultiple();
    AddInput(name.step_net, "network shared by all steps.");

    AddOutput(name.outlinks, "the outputs that need to concated for all steps.")
        .SetMultiple();
    AddOutput(name.step_scopes, "step scopes");

    // Attributes stored in AttributeMap
    AddAttr<std::vector<std::string>>(name.inlink_alias, "alias of inlinks");
    AddAttr<std::vector<std::string>>(name.outlink_alias, "alias of outlinks");
    AddAttr<std::vector<std::string>>(name.pre_memories,
                                      "names of pre-memories");
    AddAttr<std::vector<std::string>>(name.memories, "names of memories");

    AddComment("This is a recurrent group operator.");
  }
};

void RecurrentGradientAlgorithm::Run(
    const Scope& scope, const platform::DeviceContext& dev_ctx) const {
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_,
                     false /*infer_shape_mode*/);
  Variable* net = scope.FindVar(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");
  for (int step_id = seq_len_ - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len_ - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1,
                        false /*infer_shape_mode*/);
    }
    net->GetMutable<NetOp>()->Run(*step_scopes[step_id], dev_ctx);
  }
  LinkBootMemoryGradients(step_scopes[0], false);
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_,
                     false /*infer_shape_mode*/);
}

void RecurrentGradientAlgorithm::LinkBootMemoryGradients(
    Scope* step_scope, bool infer_shape_mode) const {
  for (auto& attr : arg_->memories) {
    PADDLE_ENFORCE(step_scope->FindVar(attr.var) != nullptr,
                   "memory variable [%s] does not exists", attr.var);
    PADDLE_ENFORCE(step_scope->FindVar(attr.boot_var) != nullptr,
                   "boot variable [%s] does not exists", attr.boot_var);
    Tensor* mem_grad = step_scope->NewVar(attr.var)->GetMutable<Tensor>();
    Tensor* boot_mem_grad =
        step_scope->NewVar(attr.boot_var)->GetMutable<Tensor>();
    if (infer_shape_mode) {
      boot_mem_grad->Resize(mem_grad->dims());
    } else {
      boot_mem_grad->ShareDataWith<float>(*mem_grad);
    }
  }
}

void RecurrentGradientAlgorithm::InferShape(const Scope& scope) const {
  seq_len_ = scope.FindVar((arg_->inlinks[0]).external)
                 ->GetMutable<Tensor>()
                 ->dims()[0];
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_,
                     true /*infer_shape_mode*/);
  Variable* net = scope.FindVar(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");
  for (int step_id = seq_len_ - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len_ - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1,
                        true /*infer_shape_mode*/);
    }
    net->GetMutable<NetOp>()->InferShape(*step_scopes[step_id]);
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_,
                     true /*infer_shape_mode*/);
  LinkBootMemoryGradients(step_scopes[0], true /*infer_shape_mode*/);
}

void RecurrentGradientOp::Init() {
  OperatorBase::Init();
  std::unique_ptr<rnn::Argument> arg(new rnn::Argument());
  rnn::InitArgument(kArgName, arg.get(), *this);
  alg_.Init(std::move(arg));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP(recurrent_op, paddle::operators::RecurrentOp,
            paddle::operators::RecurrentAlgorithmProtoAndCheckerMaker);
