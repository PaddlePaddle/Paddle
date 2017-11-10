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
#include "paddle/framework/executor.h"
#include "paddle/framework/op_registry.h"

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
};

class ConditionalBlockOp : public ConditionalOp {
 public:
  ConditionalBlockOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : ConditionalOp(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto xs = InputTensors(scope);
    bool need_run = std::all_of(
        xs.begin(), xs.end(),
        [](const framework::LoDTensor *t) { return t->numel() != 0; });

    if (need_run) {
      auto *scope_var = scope.FindVar(Output("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
      auto &cur_scope = *scopes->front();

      auto *block = Attr<framework::BlockDescBind *>("block");
      framework::Executor exec(dev_ctx);
      exec.Run(*block->Program(), &cur_scope, block->ID(), false);
    }
  }
};

class ConditionalBlockOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ConditionalBlockOpProtoMaker(framework::OpProto *proto,
                               framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "").AsDuplicable();
    AddInput("Params", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddOutput("Scope", "");
    AddAttr<framework::BlockDescBind *>("block", "");
    AddComment("");
  }
};

class ConditionalBlockGradOp : public ConditionalOp {
 public:
  ConditionalBlockGradOp(const std::string &type,
                         const framework::VariableNameMap &inputs,
                         const framework::VariableNameMap &outputs,
                         const framework::AttributeMap &attrs)
      : ConditionalOp(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto xs = this->InputTensors(scope);
    bool need_run = std::all_of(
        xs.begin(), xs.end(),
        [](const framework::LoDTensor *t) { return t->numel() != 0; });

    if (need_run) {
      auto *scope_var = scope.FindVar(Output("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      framework::Scope &cur_scope = *scopes[0];

      auto *block = Attr<framework::BlockDescBind *>("block");
      framework::Executor exec(dev_ctx);
      exec.Run(*block->Program(), &cur_scope, block->ID(), false);

      auto p_names = Inputs("Params");
      auto pg_names = Outputs(framework::GradVarName("Params"));

      for (size_t i = 0; i < p_names.size(); ++i) {
        auto out_grad_name = pg_names[i];
        auto in_grad_name = framework::GradVarName(p_names[i]);
        auto *in_var = cur_scope.FindVar(in_grad_name);
        if (in_var == nullptr) {
          continue;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conditional_block, ops::ConditionalBlockOp,
                  ops::ConditionalBlockOpProtoMaker);
