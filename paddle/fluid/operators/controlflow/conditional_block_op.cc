/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/controlflow/conditional_block_op.h"

#include "paddle/fluid/operators/assign_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

const char ConditionalOp::kInputs[] = "Input";
const char ConditionalOp::kOutputs[] = "Out";
const char ConditionalOp::kCondition[] = "Cond";
const char ConditionalOp::kScope[] = "Scope";
const char ConditionalOp::kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

class ConditionalBlockOp : public ConditionalOp {
 public:
  ConditionalBlockOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : ConditionalOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    bool need_run;
    if (Attr<bool>("is_scalar_condition")) {
      // When is_scalar_condition is True, the conditional variable is a scalar,
      // whether need to execute the operators in sub-block depends on the
      // conditional variable (Cond).
      auto xs = InputTensors(scope, ConditionalOp::kCondition);
      need_run = ScalarCondition(xs);
    } else {
      // When is_scalar_condition is False, the conditional variable maybe a
      // vector or tensor, whether need to execute the operators in sub-block
      // depends on the input variables (Input).
      auto xs = InputTensors(scope, ConditionalOp::kInputs);
      need_run = std::all_of(
          xs.begin(), xs.end(),
          [](const framework::LoDTensor *t) { return t->numel() != 0; });
    }

    if (need_run) {
      auto *scope_var = scope.FindVar(Output(ConditionalOp::kScope));
      PADDLE_ENFORCE_NOT_NULL(
          scope_var,
          platform::errors::PreconditionNotMet(
              "Expect Scope variable to be set in conditional_block_op, but "
              "got a null Scope variable. Please set the Scope variable."));
      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
      auto &cur_scope = *scopes->front();
      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");
      VLOG(3) << "Conditional block.idx = " << block->ID()
              << ", scope = " << &cur_scope;
      auto &skip_vars =
          Attr<std::vector<std::string>>(ConditionalOp::kSkipEagerDeletionVars);
      exec.Run(*block->Program(), &cur_scope, block->ID(), false, true,
               skip_vars, /* force_disable_gc */ false,
               /* keep_kid_scopes */ true);
    }
  }
};

class ConditionalBlockInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInputs(ConditionalOp::kCondition), true,
                      platform::errors::InvalidArgument(
                          "conditional_block_op must have condition input."));
  }
};

class ConditionalBlockGradOp : public ConditionalOp {
 public:
  ConditionalBlockGradOp(const std::string &type,
                         const framework::VariableNameMap &inputs,
                         const framework::VariableNameMap &outputs,
                         const framework::AttributeMap &attrs)
      : ConditionalOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    bool need_run;
    if (Attr<bool>("is_scalar_condition")) {
      auto xs = this->InputTensors(scope, ConditionalOp::kCondition);
      need_run = ScalarCondition(xs);
    } else {
      auto xs = this->InputTensors(scope, ConditionalOp::kInputs);
      need_run = std::all_of(
          xs.begin(), xs.end(),
          [](const framework::LoDTensor *t) { return t->numel() != 0; });
    }

    const auto &inputs = Inputs(ConditionalOp::kInputs);
    const auto &outside_grads =
        Outputs(framework::GradVarName(ConditionalOp::kInputs));
    if (need_run) {
      std::vector<std::string> inside_grads;
      inside_grads.reserve(inputs.size());
      for (auto &in : inputs) {
        inside_grads.emplace_back(framework::GradVarName(in));
      }

      auto *scope_var = scope.FindVar(Input(ConditionalOp::kScope));
      PADDLE_ENFORCE_NOT_NULL(
          scope_var,
          platform::errors::PreconditionNotMet(
              "Expect Scope variable to be set in conditional_block_op, but "
              "got a null Scope variable. Please set the Scope variable."));
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      PADDLE_ENFORCE_GT(
          scopes.size(), 0,
          platform::errors::InvalidArgument(
              "Expect Scope variable contains at least 1 scope, but got: %d",
              scopes.size()));
      framework::Scope &cur_scope = *scopes[0];

      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");

      VLOG(3) << "Conditional Grad block.idx = " << block->ID()
              << ", scope = " << &cur_scope;
      exec.Run(*block->Program(), &cur_scope, block->ID(), false, true,
               inside_grads, /* force_disable_gc */ false,
               /* keep_kid_scopes */ false);

      AssignLocalGradientToParentScope(dev_place, cur_scope, scope,
                                       inside_grads, outside_grads);
      return;
    }

    AssignZeroToParentScope(dev_place, scope, inputs, outside_grads);
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
            "Conditional block grad op doesn't support non-LoDTensor output "
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
    outside_tensor->mutable_data(place, input_tensor.dtype());
    const platform::DeviceContext *dev_ctx =
        platform::DeviceContextPool::Instance().Get(place);
    phi::funcs::set_constant(*dev_ctx, outside_tensor, 0.0f);
    outside_tensor->set_lod(input_tensor.lod());
  }
};

class ConditionalBlockGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInputs(ConditionalOp::kCondition), true,
        platform::errors::InvalidArgument(
            "Condition must be set in conditional_block_grad_op."));
    if (context->HasInputs(ConditionalOp::kInputs) &&
        context->HasOutputs(framework::GradVarName(ConditionalOp::kInputs))) {
      context->SetOutputsDim(framework::GradVarName(ConditionalOp::kInputs),
                             context->GetInputsDim(ConditionalOp::kInputs));
    }
  }
};

class ConditionalBlockGradInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    // NOTE(Aurelius84): VarType of Output is LoDTensor by default. In case of
    // Input is {Tensor, LoDTensorArray}, we need synchronous the Input's
    // VarType into Input@GRAD to avoid generating {Tensor, Tensor} as
    // Input@GRAD.
    auto input_size = ctx->InputSize(ConditionalOp::kInputs);
    auto output_size =
        ctx->OutputSize(framework::GradVarName(ConditionalOp::kInputs));
    PADDLE_ENFORCE_EQ(input_size, output_size,
                      platform::errors::InvalidArgument(
                          "input_size and output_size should be equal for "
                          "conditional_block_grad_op."));
    for (size_t i = 0; i < output_size; ++i) {
      ctx->SyncTypeAndDataType(ConditionalOp::kInputs,
                               framework::GradVarName(ConditionalOp::kInputs),
                               i);
    }
  }
};

template <typename T>
class ConditionalBlockGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("conditional_block_grad");
    grad_op->SetInput(ConditionalOp::kCondition,
                      this->Input(ConditionalOp::kCondition));
    grad_op->SetInput(ConditionalOp::kInputs,
                      this->Input(ConditionalOp::kInputs));
    grad_op->SetInput(ConditionalOp::kOutputs,
                      this->Output(ConditionalOp::kOutputs));
    grad_op->SetInput(framework::GradVarName(ConditionalOp::kOutputs),
                      this->OutputGrad(ConditionalOp::kOutputs));
    grad_op->SetInput(ConditionalOp::kScope,
                      this->Output(ConditionalOp::kScope));
    grad_op->SetOutput(framework::GradVarName(ConditionalOp::kInputs),
                       this->InputGrad(ConditionalOp::kInputs, false));
    grad_op->SetBlockAttr("sub_block", this->grad_block_[0]);
    grad_op->SetAttr("is_scalar_condition",
                     this->GetAttr("is_scalar_condition"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conditional_block, ops::ConditionalBlockOp,
                  ops::ConditionalBlockInferShape,
                  ops::ConditionalBlockOpProtoMaker,
                  ops::ConditionalBlockGradMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(conditional_block_grad, ops::ConditionalBlockGradOp,
                  ops::ConditionalBlockGradInferShape,
                  ops::ConditionalBlockGradInferVarType);
