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
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
      auto &cur_scope = *scopes->front();
      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");
      VLOG(3) << "Huihuang debug block.idx = " << block->ID()
              << ", scope = " << &cur_scope;
      auto &skip_vars =
          Attr<std::vector<std::string>>(ConditionalOp::kSkipEagerDeletionVars);
      exec.Run(*block->Program(), &cur_scope, block->ID(), false, true,
               skip_vars, /* keep_kid_scopes */ true);
      VLOG(3) << "Run scope = " << &scope;
      auto *parent = cur_scope.parent();
      while (parent) {
        VLOG(3) << "scope parent = " << parent;
        parent = parent->parent();
      }
    }
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

    if (need_run) {
      auto *scope_var = scope.FindVar(Input(ConditionalOp::kScope));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      framework::Scope &cur_scope = *scopes[0];

      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");

      const auto &inputs = Inputs(ConditionalOp::kInputs);
      const auto &outside_grads =
          Outputs(framework::GradVarName(ConditionalOp::kInputs));

      std::vector<std::string> inside_grads;
      inside_grads.reserve(inputs.size());
      for (auto &in : inputs) {
        inside_grads.emplace_back(framework::GradVarName(in));
      }

      VLOG(3) << "Huihuang debug Grad block.idx = " << block->ID()
              << ", scope = " << &cur_scope;
      VLOG(3) << "Run scope = " << &scope;
      auto *parent = cur_scope.parent();
      while (parent) {
        VLOG(3) << "scope parent = " << parent;
        parent = parent->parent();
      }

      exec.Run(*block->Program(), &cur_scope, block->ID(), false, true,
               inside_grads, /* force_disable_gc */ true,
               /* keep_kid_scopes */ true);

      AssignLocalGradientToGradScope(dev_place, cur_scope, scope, inside_grads,
                                     outside_grads);
    }
  }

 private:
  void AssignLocalGradientToGradScope(
      const platform::Place &place, const framework::Scope &cur_scope,
      const framework::Scope &parent_scope,
      const std::vector<std::string> &inside_grads,
      const std::vector<std::string> &outside_grads) const {
    for (size_t i = 0; i < outside_grads.size(); ++i) {
      const std::string &outside_grad_name = outside_grads[i];
      const std::string &inside_grad_name = inside_grads[i];
      VLOG(4) << "Huihuang gradient scope = " << &cur_scope;
      VLOG(4) << "inside_grad_name = " << inside_grad_name
              << ", outside_grad_name = " << outside_grad_name;
      framework::Variable *inside_var =
          cur_scope.FindLocalVar(inside_grad_name);
      if (inside_var == nullptr) {
        VLOG(4) << "Warning no inside var";
        continue;
      }
      framework::Variable *outside_var =
          parent_scope.FindLocalVar(outside_grad_name);
      if (outside_var == nullptr) {
        VLOG(3) << "Warning no outside var";
        continue;
      }
      const auto &tensor = inside_var->Get<framework::LoDTensor>();
      VLOG(4) << "in is initialized: " << tensor.IsInitialized();
      VLOG(3) << "tensor value " << tensor.data<float>()[0];
      platform::DeviceContext *dev_ctx =
          platform::DeviceContextPool::Instance().Get(place);
      framework::VisitVarType(*inside_var,
                              AssignFunctor(outside_var, *dev_ctx));
    }
    /*
    for (size_t i = 0; i < p_grad_names_; ++i) {
      auto out_grad_name = pg_names[i];
      const auto &in_grad_name = p_grad_names[i];
      auto *in_var = cur_scope.FindLocalVar(in_grad_name);
      VLOG(4) << "Huihuang gradient scope = " << &cur_scope;
      VLOG(4) << "in_grad_name = " << in_grad_name << ", out_grad_name = " <<
    out_grad_name;
      if (in_var == nullptr) {
        continue;
      }
      VLOG(4) << "in is initialized: " <<
    in_var->Get<framework::LoDTensor>().IsInitialized();

      auto new_in_grad_name = cur_scope.Rename(in_grad_name);

      bool cur_scope_same_out_grad_name = cur_scope.FindLocalVar(out_grad_name)
    == nullptr;
      VLOG(4) << "cur scope has same name " << cur_scope_same_out_grad_name;
      auto assign = framework::OpRegistry::CreateOp(
          "assign", {{"X", {new_in_grad_name}}}, {{"Out", {out_grad_name}}},
          framework::AttributeMap{});
      assign->Run(cur_scope, place);
      cur_scope.Rename(new_in_grad_name, in_grad_name);
    }*/
  }
};

class ConditionalBlockGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInputs(ConditionalOp::kCondition));
    // if (context->HasInputs(ConditionalOp::kInputs) &&
    // context->HasOutputs(framework::GradVarName(ConditionalOp::kInputs))) {
    //  context->SetOutputsDim(framework::GradVarName(ConditionalOp::kInputs),
    //                         context->GetInputsDim(ConditionalOp::kInputs));
    //}
  }
};

template <typename T>
class ConditionalBlockGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto grad_op = new T();
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
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conditional_block, ops::ConditionalBlockOp,
                  ops::ConditionalBlockOpProtoMaker,
                  ops::ConditionalBlockGradMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(conditional_block_grad, ops::ConditionalBlockGradOp,
                  ops::ConditionalBlockGradInferShape);
