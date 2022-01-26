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

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {

class IfBaseOp : public framework::OperatorBase {
 public:
  IfBaseOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  static const char kInputs[];
  static const char kOutputs[];
  static const char kCondition[];
  static const char kScope[];
  static const char kTrueOutVars[];
  static const char kFalseOutVars[];
  static const char kSkipEagerDeletionVars[];

 protected:
  bool IsTrueBranch(const framework::Scope &scope) const {
    if (Attr<bool>("is_scalar_condition")) {
      auto xs = this->InputTensors(scope, IfBaseOp::kCondition);
      return ScalarCondition(xs);
    } else {
      auto xs = this->InputTensors(scope, IfBaseOp::kInputs);
      return std::all_of(
          xs.begin(), xs.end(),
          [](const framework::LoDTensor *t) { return t->numel() != 0; });
    }
  }

  bool ScalarCondition(
      const std::vector<const framework::LoDTensor *> &ips) const {
    PADDLE_ENFORCE_EQ(
        ips.size() == 1UL && ips[0]->IsInitialized(), true,
        platform::errors::InvalidArgument(
            "condition should have one initialized input as condition"));

    PADDLE_ENFORCE_EQ(ips[0]->type() == framework::proto::VarType::BOOL &&
                          ips[0]->numel() == 1,
                      true, platform::errors::InvalidArgument(
                                "condition input's data type should be bool, "
                                "numel should be 1, actual numel is %d",
                                ips[0]->numel()));
    bool res = false;
    if (platform::is_gpu_place(ips[0]->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      framework::LoDTensor cpu_tensor;
      framework::TensorCopy(*ips[0], platform::CPUPlace(), &cpu_tensor);
      platform::DeviceContextPool::Instance().Get(ips[0]->place())->Wait();
      res = cpu_tensor.data<bool>()[0];
#endif
    } else if (platform::is_npu_place(ips[0]->place())) {
#ifdef PADDLE_WITH_ASCEND_CL
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

  std::vector<const framework::LoDTensor *> OutVarsFromBranch(
      const std::string &name, const framework::Scope &scope) const {
    auto inner_var_names = Attr<std::vector<std::string>>(name);
    std::vector<const framework::LoDTensor *> rets;

    for (auto &var_name : inner_var_names) {
      auto *var = scope.FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound("%s is not in scope %s.", var_name,
                                          &scope));

      auto &var_tensor = var->Get<framework::LoDTensor>();
      PADDLE_ENFORCE_EQ(var_tensor.IsInitialized(), true,
                        platform::errors::PreconditionNotMet(
                            "%s is not initialized yet.", var_name));
      rets.push_back(&var_tensor);
    }

    return rets;
  }

  std::vector<const framework::LoDTensor *> InputTensors(
      const framework::Scope &scope, const std::string &in_name) const {
    std::vector<const framework::LoDTensor *> retv;
    auto xs = Inputs(in_name);
    retv.resize(xs.size(), nullptr);
    std::transform(
        xs.begin(), xs.end(), retv.begin(),
        [&scope](const std::string &var_name) -> const framework::LoDTensor * {
          auto *var = scope.FindVar(var_name);
          PADDLE_ENFORCE_NOT_NULL(
              var, platform::errors::InvalidArgument("Cannot find variable %s",
                                                     var_name));
          return &var->Get<framework::LoDTensor>();
        });
    return retv;
  }

  // all input vars should be LoDTensor & is initialized
  void CheckVarStatus(const framework::Variable *var,
                      const std::string &var_name) const {
    PADDLE_ENFORCE_NOT_NULL(var, platform::errors::PreconditionNotMet(
                                     "%s shall not be nullptr.", var_name));
    PADDLE_ENFORCE_EQ(
        var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The input variable %s of "
            "IfOp holds "
            "wrong type. Expect type is LoDTensor, but receive type is %s.",
            var_name, platform::demangle(framework::ToTypeName(var->Type()))));
    PADDLE_ENFORCE_EQ(
        var->Get<framework::LoDTensor>().IsInitialized(), true,
        platform::errors::InvalidArgument("The tensor in input variable %s of "
                                          "If(Grad)Op "
                                          "is not initialized.",
                                          var_name));
  }

  std::vector<std::string> GetZeroGradName(
      const std::vector<std::string> &out_var_names,
      framework::Scope *outer_scope) const {
    std::vector<std::string> zero_grad_names;
    for (size_t i = 0; i < out_var_names.size(); ++i) {
      if (out_var_names[i] == framework::kEmptyVarName) continue ;
      auto *out_var = outer_scope->FindVar(out_var_names[i]); 
      PADDLE_ENFORCE_NOT_NULL(out_var, platform::errors::InvalidArgument(
          "out_var : %s is null, which is not expected. we should ensure @GRAD is created in the most_outer_scope", out_var_names[i]));
      // Don't use tensor->IsInitialized, use tensor->numel() != 0 instead. because initialized don't mean data is valid, if the numel()==0, the grad is still invalid.
      if (!(out_var->IsInitialized() &&
            out_var->Get<framework::LoDTensor>().numel() != 0)) {
        zero_grad_names.push_back(out_var_names[i]);
        VLOG(3) << "find zero_grad_var: " << out_var_names[i];
      }
    }
    return zero_grad_names;
  }
};

class IfOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(IfBaseOp::kCondition,
             "The conditional variable of this operator. If Cond is empty, the "
             "whole sub-block will not be executed.")
        .AsDuplicable();
    AddInput(IfBaseOp::kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(IfBaseOp::kOutputs, "The output variables of the sub-block.")
        .AsDuplicable();
    // TODO(Aurelius84): Need a more efficient way to replace kScope.
    AddOutput(IfBaseOp::kScope,
              "(std::vector<Scope*>) The step scope of IfOp.");
    AddAttr<framework::BlockDesc *>("true_block",
                                    "The true sub block of IfOp operator");
    AddAttr<framework::BlockDesc *>("false_block",
                                    "The false sub block of IfOp operator");
    AddAttr<bool>("is_scalar_condition",
                  "The conditional variable (Cond) is used as scalar "
                  "condition.")
        .SetDefault(false);
    AddAttr<bool>("is_grad", "whether is grad.").SetDefault(false);
    AddAttr<std::vector<std::string>>(IfBaseOp::kTrueOutVars,
                                      "Output Variable names in true sub block")
        .SetDefault(std::vector<std::string>());
    AddAttr<std::vector<std::string>>(
        IfBaseOp::kFalseOutVars, "Output Variable names in false sub block")
        .SetDefault(std::vector<std::string>());
    AddAttr<std::vector<std::string>>(IfBaseOp::kSkipEagerDeletionVars,
                                      "Vars that would not be deleted when "
                                      "garbage collection strategy enables")
        .SetDefault(std::vector<std::string>())
        .AsExtra();
    AddComment(R"DOC(If operator

If `is_scalar_condition` is True, the conditional variable (Cond) is a scalar,
run the operators in sub-block if Cond is True.

If `is_scalar_condition` is False, the conditional variable (Cond) is a vector or
tensor, run the operators in sub-block if all of input variables are not empty.


)DOC");
  }
};

}  // namespace operators
}  // namespace paddle
