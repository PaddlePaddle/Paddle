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
const char IfBaseOp::kTrueOutVars[] = "TrueOut";
const char IfBaseOp::kFalseOutVars[] = "FalseOut";
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
                                    &scope);
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
    const auto &inputs = Inputs(IfBaseOp::kInputs);
    const auto &outside_grads =
        Outputs(framework::GradVarName(IfBaseOp::kInputs));

    std::vector<std::string> inside_grads;
    inside_grads.reserve(inputs.size());
    for (auto &in : inputs) {
      inside_grads.emplace_back(framework::GradVarName(in));
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

    framework::Executor exec(dev_place);
    exec.Run(*block->Program(), &cur_scope, block->ID(), false, true,
             inside_grads, /* force_disable_gc */ false,
             /* keep_kid_scopes */ false);
    // TODO(Aurelius74): Using shareDataWith to avoid data copy.
    AssignLocalGradientToParentScope(dev_place, cur_scope, scope, inside_grads,
                                     outside_grads);
  }

 private:
  void AssignLocalGradientToParentScope(
      const platform::Place &place, const framework::Scope &cur_scope,
      const framework::Scope &parent_scope,
      const std::vector<std::string> &inside_grads,
      const std::vector<std::string> &outside_grads) const {
    for (size_t i = 0; i < outside_grads.size(); ++i) {
      const std::string &outside_grad_name = outside_grads[i];
      const std::string &inside_grad_name = inside_grads[i];
      VLOG(4) << "inside_grad_name = " << inside_grad_name
              << ", outside_grad_name = " << outside_grad_name;
      framework::Variable *inside_var =
          cur_scope.FindLocalVar(inside_grad_name);
      if (inside_var == nullptr) {
        continue;
      }
      framework::Variable *outside_var =
          parent_scope.FindVar(outside_grad_name);
      if (outside_var == nullptr) {
        continue;
      }
      platform::DeviceContext *dev_ctx =
          platform::DeviceContextPool::Instance().Get(place);
      framework::VisitVarType(*inside_var,
                              AssignFunctor(outside_var, *dev_ctx));
    }
  }

  void AssignZeroToParentScope(
      const platform::Place &place, const framework::Scope &scope,
      const std::vector<std::string> &inputs,
      const std::vector<std::string> &outside_grads) const {
    for (size_t i = 0; i < outside_grads.size(); ++i) {
      const std::string &outside_grad_name = outside_grads[i];
      const std::string &input_name = inputs[i];
      VLOG(4) << "input_name = " << input_name
              << ", outside_grad_name = " << outside_grad_name;
      framework::Variable *input_var = scope.FindVar(input_name);
      if (input_var == nullptr) {
        continue;
      }
      framework::Variable *outside_var = scope.FindVar(outside_grad_name);
      if (outside_var == nullptr) {
        continue;
      }

      if (input_var->IsType<framework::LoDTensor>()) {
        PADDLE_ENFORCE_EQ(outside_var->IsType<framework::LoDTensor>(), true,
                          platform::errors::InvalidArgument(
                              "Type of outside_var %s is NOT LoDTensor, which "
                              "doesn't match input_var %s.",
                              outside_grad_name, input_name));
        AssignZeroToOutsideTensor(
            place, scope, input_var->Get<framework::LoDTensor>(),
            outside_var->GetMutable<framework::LoDTensor>());
      } else if (input_var->IsType<framework::LoDTensorArray>()) {
        PADDLE_ENFORCE_EQ(outside_var->IsType<framework::LoDTensorArray>(),
                          true,
                          platform::errors::InvalidArgument(
                              "Type of outside_var %s is NOT LoDTensorArray, "
                              "which doesn't match input_var %s.",
                              outside_grad_name, input_name));
        const auto &input_tensors = input_var->Get<framework::LoDTensorArray>();
        auto *outside_tensors =
            outside_var->GetMutable<framework::LoDTensorArray>();
        PADDLE_ENFORCE_EQ(input_tensors.size(), outside_tensors->size(),
                          platform::errors::InvalidArgument(
                              "LoDTensorArray outside_var %s doen't have same "
                              "size as input_var %s.",
                              outside_grad_name, input_name));
        for (size_t j = 0; j < input_tensors.size(); ++j) {
          AssignZeroToOutsideTensor(place, scope, input_tensors[j],
                                    &((*outside_tensors)[j]));
        }
      } else {
        // TODO(huihuangzheng): add support for SelectedRows
        PADDLE_THROW(platform::errors::InvalidArgument(
            "IfGradop doesn't support non-LoDTensor output "
            "now."));
      }
    }
  }

  void AssignZeroToOutsideTensor(const platform::Place &place,
                                 const framework::Scope &cur_scope,
                                 const framework::LoDTensor &input_tensor,
                                 framework::LoDTensor *outside_tensor) const {
    if (!input_tensor.IsInitialized() || input_tensor.numel() == 0) {
      return;
    }
    VLOG(4) << "Assigning zero to " << outside_tensor;
    outside_tensor->Resize(input_tensor.dims());
    outside_tensor->mutable_data(place, input_tensor.type());
    const platform::DeviceContext *dev_ctx =
        platform::DeviceContextPool::Instance().Get(place);
    math::set_constant(*dev_ctx, outside_tensor, 0.0f);
    outside_tensor->set_lod(input_tensor.lod());
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
    grad_op->SetBlockAttr("sub_block", this->grad_block_[0]);
    grad_op->SetAttr("is_scalar_condition",
                     this->GetAttr("is_scalar_condition"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(if, ops::IfOp, ops::IfOpInferShape, ops::IfOpProtoMaker,
                  ops::IfGradMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(if_grad, ops::IfGradOp, ops::IfGradInferShape);
