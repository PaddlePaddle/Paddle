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

namespace paddle {
namespace operators {

using Scope = framework::Scope;
using Variable = framework::Variable;
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
    VLOG(4) << "step " << step_id << " run";
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
        VLOG(5) << "step " << i << " create " << var_name;
        step_scope.Var(var_name)->GetMutable<LoDTensor>();
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
  VLOG(10) << "---------------------------";
  auto* input0 = scope.FindVar(arg_->inlinks[0]);
  VLOG(10) << "---------------------------";
  PADDLE_ENFORCE_NOT_NULL(input0);
  VLOG(10) << "---------------------------";
  size_t seq_len = input0->GetMutable<LoDTensor>()->dims()[0];
  VLOG(10) << "---------------------------";
  auto& step_scopes = GetStepScopes(scope);
  VLOG(10) << "---------------------------";
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len);
  VLOG(10) << "---------------------------";
  for (int step_id = seq_len - 1; step_id >= 0; --step_id) {
    VLOG(10) << "---------------------------";
    if (static_cast<size_t>(step_id) != seq_len - 1) {
      rnn::LinkMemories(step_scopes, arg_->states, step_id, 1);
    }
    for (auto& op : **stepnet_) {
      op->Run(*step_scopes[step_id], dev_ctx);
    }
  }
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len, dev_ctx);
  LinkBootMemoryGradients(step_scopes[0]);
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
  alg_.Init(&arg_, &stepnet_, &vars_);
}

class RecurrentGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;
  using OpDescBind = framework::OpDescBind;

 protected:
  virtual std::unique_ptr<OpDescBind> Apply() const {
    auto* grad = new OpDescBind();
    grad->SetType(this->GradOpType());

    for (auto& input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(framework::GradVarName(input_param),
                      this->InputGrad(input_param));
    }

    for (auto& output_param : this->OutputNames()) {
      if (output_param == "step_scopes") {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->Output(output_param));
      } else {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->OutputGrad(output_param));
      }
    }

    grad->SetAttrMap(this->Attrs());

    return std::unique_ptr<OpDescBind>(grad);
  }

  virtual std::string GradOpType() const {
    return this->ForwardOpType() + "_grad";
  }
};

class RecurrentGradientOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    const auto& in_name = RecurrentOp::kArgName;
    const auto& out_name = RecurrentGradientOp::kArgName;
    PADDLE_ENFORCE(ctx->HasInput(in_name.inlinks));
    PADDLE_ENFORCE(ctx->HasInput(in_name.outlinks));
    PADDLE_ENFORCE(ctx->HasInput(out_name.inlinks));
    PADDLE_ENFORCE(ctx->HasOutput(out_name.outlinks));
    ctx->SetOutputDim(out_name.outlinks, ctx->GetInputDim(in_name.inlinks));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(recurrent, paddle::operators::RecurrentOp,
                  paddle::operators::RecurrentAlgorithmProtoAndCheckerMaker,
                  paddle::operators::RecurrentGradOpDescMaker);
REGISTER_OPERATOR(recurrent_grad, paddle::operators::RecurrentGradientOp,
                  paddle::operators::RecurrentGradientOpShapeInference);
