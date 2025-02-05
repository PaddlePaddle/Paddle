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
#include <array>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/operators/controlflow/control_flow_op_helper.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

COMMON_DECLARE_bool(use_mkldnn);

namespace paddle::operators {

const char ConditionalOp::kInputs[] = "Input";        // NOLINT
const char ConditionalOp::kOutputs[] = "Out";         // NOLINT
const char ConditionalOp::kCondition[] = "Cond";      // NOLINT
const char ConditionalOp::kScope[] = "Scope";         // NOLINT
const char ConditionalOp::kSkipEagerDeletionVars[] =  // NOLINT
    "skip_eager_deletion_vars";

using Executor = framework::Executor;
using ExecutorPrepareContext = framework::ExecutorPrepareContext;

using InterpreterCore = framework::InterpreterCore;

class ConditionalBlockOp : public ConditionalOp {
 public:
  ConditionalBlockOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : ConditionalOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const phi::Place &dev_place) const override {
    bool need_run = false;
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
      auto *scope_var = scope.FindVar(Output(ConditionalOp::kScope));
      PADDLE_ENFORCE_NOT_NULL(
          scope_var,
          common::errors::PreconditionNotMet(
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
      if (!core_ || !phi::is_same_place(core_->GetPlace(), dev_place)) {
        VLOG(10) << "[interpreterCore cache]" << core_.get();
        VLOG_IF(10, core_) << phi::is_same_place(core_->GetPlace(), dev_place);

        framework::interpreter::ExecutionConfig execution_config;
        if (HasAttr("used_for_inference") && Attr<bool>("used_for_inference")) {
          execution_config.used_for_inference = true;
        }
        execution_config.create_local_scope = false;
        execution_config.used_for_control_flow_op = true;
        execution_config.skip_gc_vars =
            std::set<std::string>(skip_vars.begin(), skip_vars.end());
        // add for performance in gpugraph transformer mode
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
        execution_config.used_for_inference = true;
#endif
        core_.reset(new InterpreterCore(
            dev_place, *block, &cur_scope, execution_config));
        core_->SetOutputHooks(output_hookfuncs_);
        core_->SetInputHooks(input_hookfuncs_);
        VLOG(10) << "[interpreterCore] created:" << core_;
      } else {
        BuildScopeForControlFlowOp(*core_, *block, &cur_scope);
        core_->reset_scope(&cur_scope);
      }

      core_->Run({}, false);
    }
  }

 private:
  mutable std::shared_ptr<Executor> exec_{nullptr};
  mutable std::unique_ptr<ExecutorPrepareContext> ctx_{nullptr};
  mutable std::shared_ptr<InterpreterCore> core_{nullptr};
};

class ConditionalBlockInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInputs(ConditionalOp::kCondition),
                      true,
                      common::errors::InvalidArgument(
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
               const phi::Place &dev_place) const override {
    bool need_run = false;
    if (Attr<bool>("is_scalar_condition")) {
      auto xs = this->InputTensors(scope, ConditionalOp::kCondition);
      need_run = ScalarCondition(xs);
    } else {
      auto xs = this->InputTensors(scope, ConditionalOp::kInputs);
      need_run =
          std::all_of(xs.begin(), xs.end(), [](const phi::DenseTensor *t) {
            return t->numel() != 0;
          });
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
          common::errors::PreconditionNotMet(
              "Expect Scope variable to be set in conditional_block_op, but "
              "got a null Scope variable. Please set the Scope variable."));
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      PADDLE_ENFORCE_GT(
          scopes.size(),
          0,
          common::errors::InvalidArgument(
              "Expect Scope variable contains at least 1 scope, but got: %d",
              scopes.size()));
      framework::Scope &cur_scope = *(scopes[0]);

      auto *block = Attr<framework::BlockDesc *>("sub_block");
      VLOG(3) << "Conditional Grad block.idx = " << block->ID()
              << ", scope = " << &cur_scope;

      LOG_FIRST_N(INFO, 1)
          << "[ControlFlow][ConditionalGradBlock] New Executor is Running.";
      if (!core_ || !phi::is_same_place(core_->GetPlace(), dev_place)) {
        VLOG(10) << "[interpreterCore cache]" << core_.get();
        VLOG_IF(10, core_) << phi::is_same_place(core_->GetPlace(), dev_place);

        framework::interpreter::ExecutionConfig execution_config;
        execution_config.create_local_scope = false;
        execution_config.used_for_control_flow_op = true;
        execution_config.skip_gc_vars =
            std::set<std::string>(inside_grads.begin(), inside_grads.end());

        core_.reset(new InterpreterCore(
            dev_place, *block, &cur_scope, execution_config));
        VLOG(10) << "[interpreterCore] created:" << core_;
      } else {
        BuildScopeForControlFlowOp(*core_, *block, &cur_scope);
        core_->reset_scope(&cur_scope);
      }
      core_->Run({}, false);

      AssignLocalGradientToParentScope(
          dev_place, cur_scope, scope, inside_grads, outside_grads, inputs);
      // Release the cur_scope, otherwise memory leakage occurs.
      scope.DeleteScope(&cur_scope);
      return;
    }

    AssignZeroToParentScope(dev_place, scope, inputs, outside_grads);
  }

 private:
  mutable std::shared_ptr<Executor> exec_{nullptr};
  mutable std::unique_ptr<ExecutorPrepareContext> ctx_{nullptr};
  mutable std::shared_ptr<InterpreterCore> core_{nullptr};
};

template <class T>
struct FilterNoGradInput {};

template <>
struct FilterNoGradInput<framework::OpDesc> {
  static void filter(const framework::BlockDesc *desc,
                     std::vector<std::string> *vec) {
    auto f = [desc](const std::string &name) -> std::string {
      if (name == framework::kEmptyVarName) {
        // don't drop empty var name, you can use Input(name, true) to drop
        // it.
        return framework::kEmptyVarName;
      }
      auto var_desc =
          desc->FindVarRecursive(framework::GradOriginalVarName(name));
      std::set<framework::proto::VarType::Type> not_support_backward_dtype = {
          framework::proto::VarType::BOOL,
          framework::proto::VarType::INT8,
          framework::proto::VarType::UINT8,
          framework::proto::VarType::INT16,
          framework::proto::VarType::INT32,
          framework::proto::VarType::INT64,
      };
      if (!var_desc ||
          not_support_backward_dtype.count(var_desc->GetDataType()))
        return framework::kEmptyVarName;
      return name;
    };
    std::transform(vec->begin(), vec->end(), vec->begin(), f);
  }
};

class ConditionalBlockGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInputs(ConditionalOp::kCondition),
        true,
        common::errors::InvalidArgument(
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
    // NOTE(Aurelius84): VarType of Output is phi::DenseTensor by default. In
    // case of Input is {Tensor, DenseTensorArray}, we need synchronous the
    // Input's VarType into Input@GRAD to avoid generating {Tensor, Tensor} as
    // Input@GRAD.
    auto input_size = ctx->InputSize(ConditionalOp::kInputs);
    auto output_size =
        ctx->OutputSize(framework::GradVarName(ConditionalOp::kInputs));
    PADDLE_ENFORCE_EQ(input_size,
                      output_size,
                      common::errors::InvalidArgument(
                          "input_size and output_size should be equal for "
                          "conditional_block_grad_op."));
    for (size_t i = 0; i < output_size; ++i) {
      ctx->SyncTypeAndDataType(ConditionalOp::kInputs,
                               framework::GradVarName(ConditionalOp::kInputs),
                               static_cast<int>(i));
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

    auto fwd_inputs = this->InputGrad(ConditionalOp::kInputs, false);
    FilterNoGradInput<T>::filter(this->GetForwardOpBlock(), &fwd_inputs);
    grad_op->SetOutput(framework::GradVarName(ConditionalOp::kInputs),
                       fwd_inputs);
    grad_op->SetBlockAttr("sub_block", this->grad_block_[0]);
    grad_op->SetAttr("is_scalar_condition",
                     this->GetAttr("is_scalar_condition"));
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;
REGISTER_OPERATOR(conditional_block,
                  ops::ConditionalBlockOp,
                  ops::ConditionalBlockInferShape,
                  ops::ConditionalBlockOpProtoMaker,
                  ops::ConditionalBlockGradMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(conditional_block_grad,
                  ops::ConditionalBlockGradOp,
                  ops::ConditionalBlockGradInferShape,
                  ops::ConditionalBlockGradInferVarType);
