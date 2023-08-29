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

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

PHI_DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

using Executor = framework::Executor;
using ExecutorPrepareContext = framework::ExecutorPrepareContext;

using InterpreterCore = framework::InterpreterCore;

class ConditionalOp : public framework::OperatorBase {
 public:
  ConditionalOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  static const char kInputs[];
  static const char kOutputs[];
  static const char kCondition[];
  static const char kScope[];
  static const char kSkipEagerDeletionVars[];

 protected:
  std::vector<const phi::DenseTensor *> InputTensors(
      const framework::Scope &scope, const std::string &in_name) const {
    std::vector<const phi::DenseTensor *> retv;
    auto xs = Inputs(in_name);
    retv.resize(xs.size(), nullptr);
    std::transform(
        xs.begin(),
        xs.end(),
        retv.begin(),
        [&scope](const std::string &var_name) -> const phi::DenseTensor * {
          auto *var = scope.FindVar(var_name);
          PADDLE_ENFORCE_NOT_NULL(var,
                                  platform::errors::InvalidArgument(
                                      "Cannot find variable %s", var_name));
          return &var->Get<phi::DenseTensor>();
        });
    return retv;
  }

  bool ScalarCondition(const std::vector<const phi::DenseTensor *> &ips) const {
    PADDLE_ENFORCE_EQ(
        ips.size() == 1UL && ips[0]->IsInitialized(),
        true,
        platform::errors::InvalidArgument(
            "condition should have one initialized input as condition"));

    PADDLE_ENFORCE_EQ(framework::TransToProtoVarType(ips[0]->dtype()) ==
                              framework::proto::VarType::BOOL &&
                          ips[0]->numel() == 1,
                      true,
                      platform::errors::InvalidArgument(
                          "condition input's data type should be bool, "
                          "numel should be 1, actual numel is %d",
                          ips[0]->numel()));
    bool res = false;
    if (platform::is_gpu_place(ips[0]->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::DenseTensor cpu_tensor;
      framework::TensorCopy(*ips[0], platform::CPUPlace(), &cpu_tensor);
      platform::DeviceContextPool::Instance().Get(ips[0]->place())->Wait();
      res = cpu_tensor.data<bool>()[0];
#endif
    } else if (platform::is_xpu_place(ips[0]->place())) {
#ifdef PADDLE_WITH_XPU
      phi::DenseTensor cpu_tensor;
      framework::TensorCopy(*ips[0], platform::CPUPlace(), &cpu_tensor);
      platform::DeviceContextPool::Instance().Get(ips[0]->place())->Wait();
      res = cpu_tensor.data<bool>()[0];
#endif
    } else if (platform::is_custom_place(ips[0]->place())) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_tensor;
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

class ConditionalBlockOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(ConditionalOp::kCondition,
             "The conditional variable of this operator. If Cond is empty, the "
             "whole sub-block will not be executed.")
        .AsDuplicable();
    AddInput(ConditionalOp::kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(ConditionalOp::kOutputs, "The output variables of the sub-block.")
        .AsDuplicable();
    AddOutput(ConditionalOp::kScope,
              "(std::vector<Scope*>) The step scope of conditional block. To "
              "unify the conditional block, rnn and while op, the type of "
              "scope is std::vector<Scope*>");
    AddAttr<framework::BlockDesc *>(
        "sub_block", "The step block of conditional block operator");
    AddAttr<bool>("is_scalar_condition",
                  "The conditional variable (Cond) is used as scalar "
                  "condition.")
        .SetDefault(false);
    AddComment(R"DOC(Conditional block operator

If `is_scalar_condition` is True, the conditional variable (Cond) is a scalar,
run the operators in sub-block if Cond is True.

If `is_scalar_condition` is False, the conditional variable (Cond) is a vector or
tensor, run the operators in sub-block if all of input variables are not empty.


)DOC");
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
      need_run =
          std::all_of(xs.begin(), xs.end(), [](const phi::DenseTensor *t) {
            return t->numel() != 0;
          });
    }

    if (need_run) {
      SetSubBlockCore(scope, dev_place);
      core_->Run({}, false);
    }
  }

 public:
  void SetSubBlockCore(const framework::Scope &scope,
                       const platform::Place &dev_place) const {
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
#ifdef PADDLE_WITH_DNNL
    // Executor on being destroyed clears oneDNN cache and resets
    // registered model data layout. This is unwanted for nested
    // Executors (executors declared inside control ops)
    platform::DontClearMKLDNNCache(dev_place);
#endif
    auto *block = Attr<framework::BlockDesc *>("sub_block");
    VLOG(3) << "Conditional block.idx = " << block->ID()
            << ", scope = " << &cur_scope;

    auto &skip_vars =
        Attr<std::vector<std::string>>(ConditionalOp::kSkipEagerDeletionVars);

    LOG_FIRST_N(INFO, 1)
        << "[ControlFlow][ConditionalBlock] New Executor is Running.";
    if (!core_ || !platform::is_same_place(core_->GetPlace(), dev_place)) {
      VLOG(10) << "[interpreterCore cache]" << core_.get();
      VLOG_IF(10, core_) << platform::is_same_place(core_->GetPlace(),
                                                    dev_place);

      framework::interpreter::ExecutionConfig execution_config;
      execution_config.create_local_scope = false;
      execution_config.used_for_control_flow_op = true;
      execution_config.skip_gc_vars =
          std::set<std::string>(skip_vars.begin(), skip_vars.end());

      core_.reset(
          new InterpreterCore(dev_place, *block, &cur_scope, execution_config));
      VLOG(10) << "[interpreterCore] created:" << core_;
    } else {
      BuildScopeForControlFlowOp(*core_, *block, &cur_scope);
      core_->reset_scope(&cur_scope);
    }
  }

  void PreStaticBuild() const { core_->PreStaticBuild(); }

 private:
  mutable std::shared_ptr<Executor> exec_{nullptr};
  mutable std::unique_ptr<ExecutorPrepareContext> ctx_{nullptr};
  mutable std::shared_ptr<InterpreterCore> core_{nullptr};
};

}  // namespace operators
}  // namespace paddle
