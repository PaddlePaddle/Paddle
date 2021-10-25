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

#include "paddle/fluid/operators/controlflow/if_op.h"

#include "glog/logging.h"
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

const char IfBaseOp::kInputs[] = "Input";
const char IfBaseOp::kOutputs[] = "Out";
const char IfBaseOp::kCondition[] = "Cond";
const char IfBaseOp::kScope[] = "Scope";
const char IfBaseOp::kTrueOutVars[] = "true_outs";
const char IfBaseOp::kFalseOutVars[] = "false_outs";
const char IfBaseOp::kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

class IfOp : public IfBaseOp {
 public:
  IfOp(const std::string &type, const framework::VariableNameMap &inputs,
       const framework::VariableNameMap &outputs,
       const framework::AttributeMap &attrs)
      : IfBaseOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &out_var_names = Outputs(IfBaseOp::kOutputs);

    bool is_true_branch = IsTrueBranch(scope);
    // Prepare scope and executor
    auto *scope_var = scope.FindVar(Output(IfOp::kScope));
    PADDLE_ENFORCE_NOT_NULL(
        scope_var,
        platform::errors::PreconditionNotMet(
            "Expect Scope variable to be set in IfOp, but "
            "got a null Scope variable. Please set the Scope variable."));
    auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
    scopes->resize(1);
    scopes->front() = &scope.NewScope();
    auto &cur_scope = *scopes->front();

    std::string branch_block_name =
        is_true_branch ? "true_block" : "false_block";
    auto *block = Attr<framework::BlockDesc *>(branch_block_name);
    VLOG(3) << "IfOp block.idx = " << block->ID() << ", scope = " << &cur_scope;
    auto &skip_vars =
        Attr<std::vector<std::string>>(IfOp::kSkipEagerDeletionVars);

    framework::Executor exec(dev_place);
    exec.Run(*block->Program(), &cur_scope, block->ID(), false, true, skip_vars,
             /* force_disable_gc */ false,
             /* keep_kid_scopes */ true);

    // Share sub_block variable to outer scope variable.
    std::string out_branch_name =
        is_true_branch ? IfBaseOp::kTrueOutVars : IfBaseOp::kFalseOutVars;
    auto &inner_var_names = Attr<std::vector<std::string>>(out_branch_name);
    ShareInnerOutVarsIntoOuterScope(inner_var_names, out_var_names, &cur_scope,
                                    const_cast<framework::Scope *>(&scope));
  }
};

class IfOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInputs(IfBaseOp::kCondition), true,
        platform::errors::InvalidArgument("IfOp must have condition input."));
  }
};

class IfGradOp : public IfBaseOp {
 public:
  IfGradOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : IfBaseOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    bool is_true_branch = IsTrueBranch(scope);
    const auto &input_names = Inputs(IfBaseOp::kInputs);
    const auto &outside_grad_names =
        Outputs(framework::GradVarName(IfBaseOp::kInputs));

    // TODO(Aurelius84): Maybe we can use outside_grad_names directly?
    std::vector<std::string> inside_grad_names;
    inside_grad_names.reserve(outside_grad_names.size());
    for (auto &name : input_names) {
      inside_grad_names.emplace_back(framework::GradVarName(name));
    }

    std::string out_branch_name =
        is_true_branch ? IfBaseOp::kTrueOutVars : IfBaseOp::kFalseOutVars;
    auto &inner_var_names = Attr<std::vector<std::string>>(out_branch_name);
    std::vector<std::string> branch_grad_names;
    for (auto &name : inner_var_names) {
      branch_grad_names.emplace_back(framework::GradVarName(name));
    }

    auto *scope_var = scope.FindVar(Input(IfBaseOp::kScope));
    PADDLE_ENFORCE_NOT_NULL(
        scope_var,
        platform::errors::PreconditionNotMet(
            "Expect Scope variable to be set in IfOp, but "
            "got a null Scope variable. Please set the Scope variable."));
    auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
    PADDLE_ENFORCE_GT(
        scopes.size(), 0,
        platform::errors::InvalidArgument(
            "Expect Scope variable contains at least 1 scope, but got: %d",
            scopes.size()));
    framework::Scope &cur_scope = *scopes[0];

    std::string branch_block_name =
        is_true_branch ? "true_block" : "false_block";
    auto *block = Attr<framework::BlockDesc *>(branch_block_name);
    VLOG(3) << "IfGrad block.idx = " << block->ID()
            << ", scope = " << &cur_scope;

    // share Out@Grad into inside_grad_names
    const auto &input_grad_names =
        Inputs(framework::GradVarName(IfBaseOp::kOutputs));
    ShareInnerOutVarsIntoOuterScope(input_grad_names, branch_grad_names, &scope,
                                    &cur_scope);

    framework::Executor exec(dev_place);
    exec.Run(*block->Program(), &cur_scope, block->ID(), false, true,
             inside_grad_names, /* force_disable_gc */ false,
             /* keep_kid_scopes */ false);

    ShareInnerOutVarsIntoOuterScope({inside_grad_names[1]},
                                    {outside_grad_names[1]}, &cur_scope,
                                    const_cast<framework::Scope *>(&scope));
    AssignZeroToOutsideTensor(dev_place, inside_grad_names[0], scope);
  }

 private:
  void AssignZeroToOutsideTensor(const platform::Place &place,
                                 const std::string &var_name,
                                 const framework::Scope &outer_scope) const {
    VLOG(4) << "Assigning zero to " << var_name;
    auto *var = outer_scope.FindVar(var_name);
    auto *outside_tensor = var->GetMutable<framework::LoDTensor>();
    outside_tensor->mutable_data(place, outside_tensor->saved_type());
    const platform::DeviceContext *dev_ctx =
        platform::DeviceContextPool::Instance().Get(place);
    math::set_constant(*dev_ctx, outside_tensor, 0.0f);
  }
};

class IfGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInputs(IfBaseOp::kCondition), true,
                      platform::errors::InvalidArgument(
                          "Condition must be set in IfGradOp."));
    if (context->HasInputs(IfBaseOp::kInputs) &&
        context->HasOutputs(framework::GradVarName(IfBaseOp::kInputs))) {
      context->SetOutputsDim(framework::GradVarName(IfBaseOp::kInputs),
                             context->GetInputsDim(IfBaseOp::kInputs));
    }
  }
};

template <typename T>
class IfGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("if_grad");
    grad_op->SetInput(IfBaseOp::kCondition, this->Input(IfBaseOp::kCondition));
    grad_op->SetInput(IfBaseOp::kInputs, this->Input(IfBaseOp::kInputs));
    grad_op->SetInput(IfBaseOp::kOutputs, this->Output(IfBaseOp::kOutputs));
    grad_op->SetInput(framework::GradVarName(IfBaseOp::kOutputs),
                      this->OutputGrad(IfBaseOp::kOutputs));
    grad_op->SetInput(IfBaseOp::kScope, this->Output(IfBaseOp::kScope));
    grad_op->SetOutput(framework::GradVarName(IfBaseOp::kInputs),
                       this->InputGrad(IfBaseOp::kInputs, false));
    grad_op->SetBlockAttr("true_block", this->grad_block_[0]);
    grad_op->SetBlockAttr("false_block", this->grad_block_[1]);
    grad_op->SetAttr("is_scalar_condition",
                     this->GetAttr("is_scalar_condition"));
    grad_op->SetAttr(IfBaseOp::kTrueOutVars,
                     this->GetAttr(IfBaseOp::kTrueOutVars));
    grad_op->SetAttr(IfBaseOp::kFalseOutVars,
                     this->GetAttr(IfBaseOp::kFalseOutVars));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(if, ops::IfOp, ops::IfOpInferShape, ops::IfOpProtoMaker,
                  ops::IfGradMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(if_grad, ops::IfGradOp, ops::IfGradInferShape);
