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

#include <algorithm>

#include "glog/logging.h"
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

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
               const platform::Place &place) const override {
    // Step 1. Prepare pred/Input/Output
    bool is_true_branch = IsTrueBranch(scope);
    bool is_grad = Attr<bool>("is_grad");
    auto &out_names = Outputs(IfBaseOp::kOutputs);

    std::string branch_name =
        is_true_branch ? IfBaseOp::kTrueOutVars : IfBaseOp::kFalseOutVars;
    std::string branch_block_name =
        is_true_branch ? "true_block" : "false_block";
    auto *block = Attr<framework::BlockDesc *>(branch_block_name);
    VLOG(3) << "IfOp block.idx = " << block->ID();
    auto &skip_vars =
        Attr<std::vector<std::string>>(IfOp::kSkipEagerDeletionVars);

    // Step 2. Prepare scope
    auto *scope_var = scope.FindVar(Output(IfOp::kScope));
    PADDLE_ENFORCE_NOT_NULL(
        scope_var,
        platform::errors::PreconditionNotMet(
            "Expect Scope variable to be set in IfOp, but "
            "got a null Scope variable. Please set the Scope variable."));
    auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
    if (!is_grad) {
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
    }
    PADDLE_ENFORCE_EQ(
        scopes->size(), 1U,
        platform::errors::PreconditionNotMet(
            "Expected scopes.size() == 1, but received %d .", scopes->size()));
    auto &cur_scope = *scopes->front();

    if (is_grad) {
      VLOG(5) << "If curscope is : " << GenScopeTreeDebugInfo(&cur_scope);
    }
    // Step 3. Prepare Executor and Execute it.
    framework::Executor exec(place);
    exec.Run(*block->Program(), &cur_scope, block->ID(), false, true, skip_vars,
             /* force_disable_gc */ false,
             /* keep_kid_scopes */ !is_grad);

    // Step 4. Share into outer scope.
    if (is_grad) {
      auto zero_grad_names =
          GetZeroGradName(out_names, const_cast<framework::Scope *>(&scope));
      AssignZeroToOutsideTensor(place, zero_grad_names, scope);
    }
  }

  void AssignZeroToOutsideTensor(const platform::Place &place,
                                 const std::vector<std::string> &var_names,
                                 const framework::Scope &outer_scope) const {
    for (auto &var_name : var_names) {
      VLOG(4) << "Assigning zero to " << var_name;
      auto *var = outer_scope.FindVar(var_name);
      // NOTE(Aurelius84): how to know the ddims
      // NOTE(xiongkun03): use the GradOriginalVarName() to get the ddims.
      auto *outside_tensor = var->GetMutable<framework::LoDTensor>();
      auto &ddim =
          outer_scope.FindVar(framework::GradOriginalVarName(var_name))
              ->Get<framework::LoDTensor>()
              .dims();
      outside_tensor->Resize(ddim);
      outside_tensor->mutable_data(place, outside_tensor->type());
      const platform::DeviceContext *dev_ctx =
          platform::DeviceContextPool::Instance().Get(place);
      phi::funcs::set_constant(*dev_ctx, outside_tensor, 0.0f);
      VLOG(4) << "the number of " << var_name
              << "is: " << outside_tensor->numel();
    }
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
    grad_op->SetType("if");
    grad_op->SetInput(IfBaseOp::kCondition, this->Input(IfBaseOp::kCondition));
    // [x, y, out1, out2, out1@GRAD, out2@GRAD]
    auto input_names = this->Input(IfBaseOp::kInputs);
    auto input_names_copy = input_names;
    auto out_names = this->Output(IfBaseOp::kOutputs);
    auto out_grad_name = this->OutputGrad(IfBaseOp::kOutputs);
    input_names.insert(input_names.end(), out_names.begin(), out_names.end());
    input_names.insert(input_names.end(), out_grad_name.begin(),
                       out_grad_name.end());
    grad_op->SetInput(IfBaseOp::kInputs, input_names);

    grad_op->SetOutput(IfBaseOp::kScope, this->Output(IfBaseOp::kScope));
    grad_op->SetOutput(IfBaseOp::kOutputs,
                       this->InputGrad(IfBaseOp::kInputs, false));
    grad_op->SetBlockAttr("true_block", this->grad_block_[0]);
    grad_op->SetBlockAttr("false_block", this->grad_block_[1]);
    grad_op->SetAttr("is_scalar_condition",
                     this->GetAttr("is_scalar_condition"));
    grad_op->SetAttr("is_grad", true);

    std::vector<std::string> false_out_grad_names;
    std::for_each(input_names_copy.begin(), input_names_copy.end(),
                  [&false_out_grad_names](std::string &name) {
                    false_out_grad_names.emplace_back(name + "@GRAD");
                  });
    std::vector<std::string> true_out_grad_names = false_out_grad_names;
    grad_op->SetAttr(IfBaseOp::kTrueOutVars, false_out_grad_names);  // the same
    grad_op->SetAttr(IfBaseOp::kFalseOutVars, true_out_grad_names);
    grad_op->SetAttr(IfBaseOp::kSkipEagerDeletionVars,
                     this->InputGrad(IfBaseOp::kInputs, false));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(if, ops::IfOp, ops::IfOpInferShape, ops::IfOpProtoMaker,
                  ops::IfGradMaker<paddle::framework::OpDesc>);
