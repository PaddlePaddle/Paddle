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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

using Scope = framework::Scope;
using Variable = framework::Variable;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

void RecurrentAlgorithm::Run(const Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  auto* input0 = scope.FindVar(arg_->inlinks[0]);
  PADDLE_ENFORCE_NOT_NULL(input0);
  size_t seq_len = input0->GetMutable<LoDTensor>()->dims()[0];
  PADDLE_ENFORCE_GT(seq_len, 0);

  CreateScopes(scope, seq_len);
  auto& step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len);
  InitMemories(step_scopes[0]);

  for (size_t step_id = 0; step_id < seq_len; step_id++) {
    LOG(INFO) << "step " << step_id << " run";
    if (step_id > 0) {
      rnn::LinkMemories(step_scopes, arg_->states, step_id, -1);
    }
    for (auto& op : **stepnet_) {
      op->Run(*step_scopes[step_id], dev_ctx);
    }
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len, dev_ctx);
}

void RecurrentAlgorithm::CreateScopes(const Scope& scope,
                                      size_t seq_len) const {
  // TODO(superjom) Only two scopes are needed for inference, this case will be
  // supported later.
  auto* step_scopes_var = scope.FindVar(arg_->step_scopes);
  PADDLE_ENFORCE(step_scopes_var != nullptr, "");
  auto* step_scopes = step_scopes_var->GetMutable<std::vector<Scope*>>();

  // Now all variables in scope must be created outside of op.
  PADDLE_ENFORCE_NOT_NULL(stepnet_);
  PADDLE_ENFORCE_NOT_NULL(vars_);
  PADDLE_ENFORCE(!(*stepnet_)->empty());

  if (seq_len > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len; ++i) {
      auto& step_scope = scope.NewScope();
      for (auto& var_name : *vars_) {
        LOG(INFO) << "step " << i << " create " << var_name;
        step_scope.Var(var_name);
      }
      step_scopes->emplace_back(&step_scope);
    }
  }
}

void RecurrentAlgorithm::InitMemories(Scope* step_scope) const {
  for (auto& attr : arg_->states) {
    auto* pre_mem = step_scope->Var(attr.pre_var)->GetMutable<LoDTensor>();
    PADDLE_ENFORCE(step_scope->FindVar(attr.boot_var) != nullptr,
                   "memory [%s]'s boot variable [%s] not exists", attr.var,
                   attr.boot_var);
    auto* boot_mem =
        step_scope->FindVar(attr.boot_var)->GetMutable<LoDTensor>();
    pre_mem->Resize(boot_mem->dims());
    PADDLE_ENFORCE_EQ(pre_mem->dims().size(), 2);
    pre_mem->ShareDataWith(*boot_mem);
  }
}

const rnn::ArgumentName RecurrentOp::kArgName{
    "step_net", "step_scopes", "inputs",        "outputs",
    "states",   "ex_states",   "initial_states"};

const rnn::ArgumentName RecurrentGradientOp::kArgName{
    "step_net", "step_scopes@GRAD", "outputs@GRAD",       "inputs@GRAD",
    "states",   "ex_states",        "initial_states@GRAD"};

RecurrentOp::RecurrentOp(const std::string& type,
                         const framework::VariableNameMap& inputs,
                         const framework::VariableNameMap& outputs,
                         const framework::AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs) {
  rnn::InitArgument(kArgName, &arg_, *this);
  alg_.Init(&arg_, &stepnet_, &vars_);
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
    AddInput(name.initial_states, "variables to initialize states.")
        .AsDuplicable();

    AddInput("parameters", "parameter variables used inside").AsDuplicable();

    AddOutput(name.outlinks, "the outputs that need to concated for all steps.")
        .AsDuplicable();
    AddOutput(name.step_scopes, "step scopes");

    // Attributes stored in AttributeMap
    AddAttr<std::vector<std::string>>(name.ex_states, "names of pre-states");
    AddAttr<std::vector<std::string>>(name.states, "names of states");
    AddAttr<paddle::framework::BlockDesc*>("block_idx", "rnn block idx");

    AddComment("This is a recurrent group operator.");
  }
};

void RecurrentGradientAlgorithm::Run(
    const Scope& scope, const platform::DeviceContext& dev_ctx) const {
  auto* input0 = scope.FindVar(arg_->inlinks[0]);
  PADDLE_ENFORCE_NOT_NULL(input0);
  size_t seq_len = input0->GetMutable<LoDTensor>()->dims()[0];
  auto& step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len);
  CreateParamLocalGradients(step_scopes, seq_len);
  for (int step_id = seq_len - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len - 1) {
      rnn::LinkMemories(step_scopes, arg_->states, step_id, 1);
    }
    for (auto& op : **stepnet_) {
      op->Run(*step_scopes[step_id], dev_ctx);
    }
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len, dev_ctx);
  LinkBootMemoryGradients(step_scopes[0]);
  ExposeWeightGradients(step_scopes, seq_len, dev_ctx);
}

void RecurrentGradientAlgorithm::LinkBootMemoryGradients(
    Scope* step_scope) const {
  for (auto& attr : arg_->states) {
    PADDLE_ENFORCE(step_scope->FindVar(attr.var) != nullptr,
                   "memory variable [%s] does not exists", attr.var);
    PADDLE_ENFORCE(step_scope->FindVar(attr.boot_var) != nullptr,
                   "boot variable [%s] does not exists", attr.boot_var);
    auto* mem_grad = step_scope->Var(attr.var)->GetMutable<LoDTensor>();
    auto* boot_mem_grad =
        step_scope->Var(attr.boot_var)->GetMutable<LoDTensor>();
    boot_mem_grad->Resize(mem_grad->dims());
    boot_mem_grad->ShareDataWith(*mem_grad);
  }
}

RecurrentGradientOp::RecurrentGradientOp(
    const std::string& type, const framework::VariableNameMap& inputs,
    const framework::VariableNameMap& outputs,
    const framework::AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs) {
  rnn::InitArgument(kArgName, &arg_, *this, true /*is grad*/);
  alg_.Init(&arg_, &stepnet_, &vars_, Inputs("parameters"));
}

void RecurrentGradientAlgorithm::CreateParamLocalGradients(
    const std::vector<Scope*>& step_scopes, size_t seq_len) const {
  for (const std::string& param_name : parameters_) {
    for (size_t step = 0; step < seq_len; step++) {
      step_scopes[step]->Var(param_name);
    }
  }
}

void RecurrentGradientAlgorithm::ExposeWeightGradients(
    const std::vector<framework::Scope*>& step_scopes, size_t seq_len,
    const platform::DeviceContext& dev_ctx) const {
  // NOTE addto the gradients in global scope is not thread-safe
  // sum the gradients in all the step_scopes
  // output to the global scope

  auto& parent_scope = step_scopes.front()->parent();

  for (auto& param : parameters_) {
    // TODO(superjom) Make sure that parameter names will be transformed to
    // gradient name.
    // std::string param_grad_name = framework::GradVarName(param);
    std::string param_grad_name = param;
    framework::LoDTensor& param_gradient =
        *parent_scope.FindVar(param_grad_name)->GetMutable<LoDTensor>();
    auto param_gradient_eigen =
        framework::EigenVector<float>::Flatten(param_gradient);

    auto place = dev_ctx.GetEigenDevice<platform::CPUPlace>();
    for (size_t step = 0; step < seq_len; step++) {
      Tensor& step_param_gradient =
          *step_scopes[step]->Var(param_grad_name)->GetMutable<LoDTensor>();
      auto step_param_gradient_eigen =
          framework::EigenVector<float>::Flatten(step_param_gradient);
      param_gradient_eigen.device(*place) =
          step_param_gradient_eigen + param_gradient_eigen;
    }
  }
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP(recurrent, paddle::operators::RecurrentOp,
            paddle::operators::RecurrentAlgorithmProtoAndCheckerMaker,
            recurrent_grad, paddle::operators::RecurrentGradientOp);
