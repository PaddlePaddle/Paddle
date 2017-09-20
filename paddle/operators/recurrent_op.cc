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

#include <cstring>
#include <sstream>

#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

using Scope = framework::Scope;
using Variable = framework::Variable;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

void RecurrentAlgorithm::InferShape(const Scope& scope) const {
  seq_len_ = scope.FindVar((arg_->inlinks[0]).external)
                 ->GetMutable<LoDTensor>()
                 ->dims()[0];
  CreateScopes(scope);
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_,
                     true /*infer_shape_mode*/);
  InitMemories(step_scopes[0], true /*infer_shape_mode*/);

  for (size_t i = 0; i < seq_len_; i++) {
    if (i > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, i, -1,
                        true /*infer_shape_mode*/);
    }
    (*stepnet_)->InferShape(*step_scopes[i]);
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

  for (size_t step_id = 0; step_id < seq_len_; step_id++) {
    // create output alias variables
    if (step_id > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, -1,
                        false /*infer_shape_mode*/);
    }
    (*stepnet_)->Run(*step_scopes[step_id], dev_ctx);
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_,
                     false /*infer_shape_mode*/);
}

void RecurrentAlgorithm::CreateScopes(const Scope& scope) const {
  // TODO(superjom) Only two scopes are needed for inference, this case will be
  // supported later.
  auto step_scopes_var = scope.FindVar(arg_->step_scopes);
  PADDLE_ENFORCE(step_scopes_var != nullptr, "");
  auto step_scopes = step_scopes_var->GetMutable<std::vector<Scope*>>();

  // Now all variables in scope must be created outside of op.
  PADDLE_ENFORCE_NOT_NULL(stepnet_);
  PADDLE_ENFORCE(!(*stepnet_)->Outputs().empty(), "stepnet_ op has no outputs");
  PADDLE_ENFORCE(!(*stepnet_)->Outputs().empty(), "net_op has no outputs");

  if (seq_len_ > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len_; ++i) {
      auto& step_scope = scope.NewScope();

      // create step net's temp inputs
      for (auto& input : (*stepnet_)->Inputs()) {
        // the weight are located in parent scope
        for (auto& var_name : input.second) {
          if (!step_scope.FindVar(var_name)) {
            step_scope.NewVar(var_name)->GetMutable<LoDTensor>();
          }
        }
      }
      // create stepnet's outputs
      for (const auto& output : (*stepnet_)->Outputs()) {
        for (auto& var_name : output.second) {
          step_scope.NewVar(var_name);
        }
      }
      step_scopes->emplace_back(&step_scope);
    }
  }
}

void RecurrentAlgorithm::InitMemories(Scope* step_scope,
                                      bool infer_shape_mode) const {
  for (auto& attr : arg_->memories) {
    auto* pre_mem = step_scope->NewVar(attr.pre_var)->GetMutable<LoDTensor>();
    PADDLE_ENFORCE(step_scope->FindVar(attr.boot_var) != nullptr,
                   "memory [%s]'s boot variable [%s] not exists", attr.var,
                   attr.boot_var);
    auto* boot_mem =
        step_scope->FindVar(attr.boot_var)->GetMutable<LoDTensor>();
    if (infer_shape_mode) {
      pre_mem->Resize(boot_mem->dims());
      PADDLE_ENFORCE_EQ(pre_mem->dims().size(), 2);
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

RecurrentOp::RecurrentOp(const std::string& type,
                         const framework::VariableNameMap& inputs,
                         const framework::VariableNameMap& outputs,
                         const framework::AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs) {
  rnn::InitArgument(kArgName, &arg_, *this);
  alg_.Init(&arg_, &stepnet_);
}

class RecurrentAlgorithmProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  RecurrentAlgorithmProtoAndCheckerMaker(framework::OpProto* proto,
                                         framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    const auto& name = RecurrentOp::kArgName;
    // inputs and outputs stored in proto
    AddInput(name.inlinks,
             "the inputs that need to be segmented for each step.")
        .AsDuplicable();
    AddInput(name.boot_memories, "variables to initialize memories.")
        .AsDuplicable();

    AddOutput(name.outlinks, "the outputs that need to concated for all steps.")
        .AsDuplicable();
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
  for (int step_id = seq_len_ - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len_ - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1,
                        false /*infer_shape_mode*/);
    }
    (*stepnet_)->Run(*step_scopes[step_id], dev_ctx);
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
    auto* mem_grad = step_scope->NewVar(attr.var)->GetMutable<LoDTensor>();
    auto* boot_mem_grad =
        step_scope->NewVar(attr.boot_var)->GetMutable<LoDTensor>();
    if (infer_shape_mode) {
      boot_mem_grad->Resize(mem_grad->dims());
    } else {
      boot_mem_grad->ShareDataWith<float>(*mem_grad);
    }
  }
}

void RecurrentGradientAlgorithm::InferShape(const Scope& scope) const {
  seq_len_ = scope.FindVar((arg_->inlinks[0]).external)
                 ->GetMutable<LoDTensor>()
                 ->dims()[0];
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_,
                     true /*infer_shape_mode*/);
  for (int step_id = seq_len_ - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len_ - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1,
                        true /*infer_shape_mode*/);
    }
    (*stepnet_)->InferShape(*step_scopes[step_id]);
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_,
                     true /*infer_shape_mode*/);
  LinkBootMemoryGradients(step_scopes[0], true /*infer_shape_mode*/);
}

RecurrentGradientOp::RecurrentGradientOp(
    const std::string& type, const framework::VariableNameMap& inputs,
    const framework::VariableNameMap& outputs,
    const framework::AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs) {
  rnn::InitArgument(kArgName, &arg_, *this);
  alg_.Init(&arg_, &stepnet_);
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    recurrent, paddle::operators::RecurrentOp,
    paddle::operators::RecurrentAlgorithmProtoAndCheckerMaker);
