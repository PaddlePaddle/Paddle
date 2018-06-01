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
#include <algorithm>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ConditionalOp : public framework::OperatorBase {
 public:
  ConditionalOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 protected:
  std::vector<const framework::LoDTensor *> InputTensors(
      const framework::Scope &scope) const {
    std::vector<const framework::LoDTensor *> retv;
    auto xs = Inputs("X");
    retv.resize(xs.size(), nullptr);
    std::transform(
        xs.begin(), xs.end(), retv.begin(),
        [&scope](const std::string &var_name) -> const framework::LoDTensor * {
          auto *var = scope.FindVar(var_name);
          PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s", var_name);
          return &var->Get<framework::LoDTensor>();
        });
    return retv;
  }

  bool ScalarCondition(
      const std::vector<const framework::LoDTensor *> &ips) const {
    if (!(ips.size() == 1UL && ips[0]->IsInitialized())) {
      PADDLE_THROW("should have one initialized input as condition");
    }
    if (!(ips[0]->type().hash_code() == typeid(bool).hash_code() &&  // NOLINT
          ips[0]->numel() == 1)) {
      PADDLE_THROW(
          "condition input's data type should be bool, "
          "numel should be 1, actual numel is %d",
          ips[0]->numel());
    }
    bool res = false;
    if (platform::is_gpu_place(ips[0]->place())) {
#ifdef PADDLE_WITH_CUDA
      framework::LoDTensor cpu_tensor;
      framework::TensorCopy(*ips[0], platform::CPUPlace(), &cpu_tensor);
      platform::DeviceContextPool::Instance().Get(ips[0]->place())->Wait();
      res = cpu_tensor.data<bool>()[0];
#endif
    } else {
      res = ips[0]->data<bool>()[0];
    }
    return res;
  }
};

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
    auto xs = InputTensors(scope);

    bool need_run;
    if (Attr<bool>("is_scalar_condition")) {
      need_run = ScalarCondition(xs);
    } else {
      need_run = std::all_of(
          xs.begin(), xs.end(),
          [](const framework::LoDTensor *t) { return t->numel() != 0; });
    }

    if (need_run) {
      auto *scope_var = scope.FindVar(Output("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
      auto &cur_scope = *scopes->front();

      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");
      exec.Run(*block->Program(), &cur_scope, block->ID(), false);
    }
  }
};

class ConditionalBlockOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The conditional variable of this operator. If X is empty, the "
             "whole sub-block will not be executed.")
        .AsDuplicable();
    AddInput("Params", "The input variables of the sub-block.").AsDuplicable();
    AddOutput("Out", "The output variables of the sub-block.").AsDuplicable();
    AddOutput("Scope",
              "(std::vector<Scope*>) The step scope of conditional block. To "
              "unify the conditional block, rnn and while op, the type of "
              "scope is std::vector<Scope*>");
    AddAttr<framework::BlockDesc *>(
        "sub_block", "The step block of conditional block operator");
    AddAttr<bool>("is_scalar_condition",
                  "the input X is used as scalar "
                  "condition")
        .SetDefault(false);
    AddComment(R"DOC(Conditional block operator

Run the sub-block if X is not empty. Params is the other inputs and Out is the
outputs of the sub-block.
)DOC");
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
    auto xs = this->InputTensors(scope);

    bool need_run;
    if (Attr<bool>("is_scalar_condition")) {
      need_run = ScalarCondition(xs);
    } else {
      need_run = std::all_of(
          xs.begin(), xs.end(),
          [](const framework::LoDTensor *t) { return t->numel() != 0; });
    }

    if (need_run) {
      auto *scope_var = scope.FindVar(Input("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      framework::Scope &cur_scope = *scopes[0];

      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");
      exec.Run(*block->Program(), &cur_scope, block->ID(), false);

      AssignLocalGradientToGlobal(dev_place, cur_scope, Inputs("Params"),
                                  Outputs(framework::GradVarName("Params")));

      AssignLocalGradientToGlobal(dev_place, cur_scope, Inputs("X"),
                                  Outputs(framework::GradVarName("X")));
    }
  }

 private:
  void AssignLocalGradientToGlobal(
      const platform::Place &place, const framework::Scope &cur_scope,
      const std::vector<std::string> &p_names,
      const std::vector<std::string> &pg_names) const {
    for (size_t i = 0; i < p_names.size(); ++i) {
      auto out_grad_name = pg_names[i];
      auto in_grad_name = framework::GradVarName(p_names[i]);
      auto *in_var = cur_scope.FindVar(in_grad_name);
      if (in_var == nullptr) {
        continue;
      }
      auto new_in_grad_name = cur_scope.Rename(in_grad_name);
      auto assign = framework::OpRegistry::CreateOp(
          "assign", {{"X", {new_in_grad_name}}}, {{"Out", {out_grad_name}}},
          framework::AttributeMap{});
      assign->Run(cur_scope, place);
      cur_scope.Rename(new_in_grad_name, in_grad_name);
    }
  }
};

class ConditionalBlockGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInputs("X"));
    if (context->HasInputs("Params")) {
      PADDLE_ENFORCE(context->HasOutputs(framework::GradVarName("Params")));
      context->SetOutputsDim(framework::GradVarName("Params"),
                             context->GetInputsDim("Params"));
    }
    PADDLE_ENFORCE(context->HasOutputs(framework::GradVarName("X")));
    context->SetOutputsDim(framework::GradVarName("X"),
                           context->GetInputsDim("X"));
  }
};

class ConditionalBlockGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto grad_op = new framework::OpDesc();
    grad_op->SetType("conditional_block_grad");
    grad_op->SetInput("X", Input("X"));
    grad_op->SetInput("Params", Input("Params"));
    grad_op->SetInput("Out", Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetInput("Scope", Output("Scope"));
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X", false));
    grad_op->SetOutput(framework::GradVarName("Params"),
                       InputGrad("Params", false));
    grad_op->SetBlockAttr("sub_block", this->grad_block_[0]);
    grad_op->SetAttr("is_scalar_condition", GetAttr("is_scalar_condition"));
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conditional_block, ops::ConditionalBlockOp,
                  ops::ConditionalBlockOpProtoMaker,
                  ops::ConditionalBlockGradMaker);
REGISTER_OPERATOR(conditional_block_grad, ops::ConditionalBlockGradOp,
                  ops::ConditionalBlockGradInferShape);
