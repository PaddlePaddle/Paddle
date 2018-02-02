/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/executor.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class SwitchOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SwitchOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The conditional variable of this operator.").AsDuplicable();
    AddOutput("Scope",
              "(std::vector<Scope*>) The step scope of conditional block. To "
              "unify the conditional block, rnn and while op, "
              "the type of scope is std::vector<Scope*>");
    AddAttr<std::vector<framework::BlockDesc *>>(
        "sub_blocks",
        "The step block of conditional "
        "block operator, the length should be the same as X");
    AddComment(R"DOC(switch operator

Run one sub block according to condition list,
)DOC");
  }
};

class SwitchOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto grad_op = new framework::OpDesc();
    grad_op->SetType("switch_grad");
    grad_op->SetInput("X", Input("X"));
    grad_op->SetInput("Scope", Output("Scope"));
    grad_op->SetBlocksAttr("sub_blocks", this->grad_block_);
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class SwitchOpBase : public framework::OperatorBase {
 public:
  SwitchOpBase(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

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

  int GetMatchCaseIndex(
      const std::vector<const framework::LoDTensor *> &conditions,
      const std::vector<framework::BlockDesc *> &case_blocks) const {
    size_t cond_num = conditions.size();
    size_t case_num = case_blocks.size();

    int match_cond_id = -1;

    for (size_t i = 0; i < conditions.size(); ++i) {
      auto cond = conditions[i];
      PADDLE_ENFORCE(cond->IsInitialized() &&
                         cond->dims() == framework::make_ddim({1}) &&
                         cond->type().hash_code() == typeid(bool).hash_code(),
                     "cond should be a scalar bool tensor");
      if (cond->data<bool>()[0]) {
        match_cond_id = static_cast<int>(i);
        break;
      }
    }

    if (match_cond_id >= 0) {
      return match_cond_id;
    } else if (cond_num + 1 == case_num) {
      return case_num - 1;
    } else {
      return -1;
    }
  }
};

class SwitchOp : public SwitchOpBase {
 public:
  SwitchOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : SwitchOpBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::Place &dev_place) const override {
    auto xs = InputTensors(scope);
    auto blocks = Attr<std::vector<framework::BlockDesc *>>("sub_blocks");

    size_t cond_num = xs.size();
    size_t case_num = blocks.size();
    PADDLE_ENFORCE(cond_num == case_num || cond_num + 1 == case_num,
                   "cond_num %d and case_num %d does not meet requirement",
                   cond_num, case_num);
    int match_case_id = GetMatchCaseIndex(xs, blocks);
    if (match_case_id >= 0) {
      VLOG(3) << "match case " << match_case_id;
      auto block = blocks[match_case_id];

      auto *scope_var = scope.FindVar(Output("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
      auto &cur_scope = *scopes->front();

      framework::Executor exec(dev_place);

      exec.Run(*block->Program(), &cur_scope, block->ID(), false);
    } else {
      VLOG(3) << "no case is matched, do nothing";
    }
  }
};

class SwitchGradOp : public SwitchOpBase {
 public:
  SwitchGradOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : SwitchOpBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::Place &dev_place) const override {
    auto xs = this->InputTensors(scope);
    bool need_run = std::all_of(
        xs.begin(), xs.end(),
        [](const framework::LoDTensor *t) { return t->numel() != 0; });

    if (need_run) {
      auto *scope_var = scope.FindVar(Input("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      framework::Scope &cur_scope = *scopes[0];

      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_blocks");
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

class SwitchGradInferShape : public framework::InferShapeBase {
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(switch, ops::SwitchOp, ops::SwitchOpProtoMaker,
                  ops::SwitchOpGradMaker);
REGISTER_OPERATOR(switch_grad, ops::SwitchGradOp, ops::SwitchGradInferShape);
