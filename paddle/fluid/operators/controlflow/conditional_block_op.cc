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

#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/fluid/operators/controlflow/control_flow_op_helper.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

const char ConditionalOp::kInputs[] = "Input";
const char ConditionalOp::kOutputs[] = "Out";
const char ConditionalOp::kCondition[] = "Cond";
const char ConditionalOp::kScope[] = "Scope";
const char ConditionalOp::kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

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
      auto *scope_var = scope.FindVar(Output(ConditionalOp::kScope));
      PADDLE_ENFORCE_NOT_NULL(
          scope_var,
          platform::errors::PreconditionNotMet(
              "Expect Scope variable to be set in conditional_block_op, but "
              "got a null Scope variable. Please set the Scope variable."));

      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();

      if (scopes->size() == 0 || !FLAGS_control_flow_use_new_executor) {
        scopes->resize(1);
        scopes->front() = &scope.NewScope();
      }

      // We need to know whether the scope we cached is still valid.
      // If not, we need to create a new one.
      if (scope.kids().size() == 0) {
        scopes->front() = &scope.NewScope();
      }

      auto &cur_scope = *scopes->front();
#ifdef PADDLE_WITH_MKLDNN
      // (jczaja) Executor on being destroyed clears oneDNN cache and
      // reset registered model data layout. This is unwanted for nested
      // Executors (executors declared inside control ops)
      platform::DontClearMKLDNNCache(dev_place);
#endif
      auto *block = Attr<framework::BlockDesc *>("sub_block");
      VLOG(3) << "Conditional block.idx = " << block->ID()
              << ", scope = " << &cur_scope;

      auto &skip_vars =
          Attr<std::vector<std::string>>(ConditionalOp::kSkipEagerDeletionVars);

      if (FLAGS_control_flow_use_new_executor) {
        LOG_FIRST_N(INFO, 1)
            << "[ControlFlow][ConditionalBlock] New Executor is Running.";
        if (!core_ || !platform::is_same_place(core_->GetPlace(), dev_place)) {
          std::set<std::string> skip_gc_vars(skip_vars.begin(),
                                             skip_vars.end());
          VLOG(10) << "[interpreterCore cache]" << core_.get();
          VLOG_IF(10, core_)
              << platform::is_same_place(core_->GetPlace(), dev_place);
          core_.reset(new InterpreterCore(dev_place,
                                          *block,
                                          skip_gc_vars,
                                          &cur_scope,
                                          /* used_for_jit */ false,
                                          /* used_for_control_flow_op */ true));
          VLOG(10) << "[interpreterCore cache]"
                   << "new created:" << core_;
        } else {
          BuildScopeForControlFlowOp(*core_, *block, &cur_scope);
          core_->reset_scope(&cur_scope);
        }

        core_->Run({}, false);

      } else {
        if (!exec_ || !platform::is_same_place(exec_->GetPlace(), dev_place)) {
          auto &pdesc = *block->Program();
          exec_.reset(new Executor(dev_place));
          if (FLAGS_use_mkldnn) exec_->EnableMKLDNN(pdesc);
          ctx_ = exec_->Prepare(pdesc, block->ID(), skip_vars, false);
#ifdef PADDLE_WITH_MKLDNN
          platform::AttachPointerHashToMKLDNNKey(exec_.get(), dev_place);
          platform::RegisterModelLayout(ctx_->ops_, dev_place);
#endif
        }
        exec_->RunPreparedContext(ctx_.get(),
                                  &cur_scope,
                                  /* create_local_scope */ false,
                                  /* create_vars */ true,
                                  /* keep_kids */ true);
      }
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
          platform::errors::PreconditionNotMet(
              "Expect Scope variable to be set in conditional_block_op, but "
              "got a null Scope variable. Please set the Scope variable."));
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      PADDLE_ENFORCE_GT(
          scopes.size(),
          0,
          platform::errors::InvalidArgument(
              "Expect Scope variable contains at least 1 scope, but got: %d",
              scopes.size()));
      framework::Scope &cur_scope = *(scopes[0]);

      auto *block = Attr<framework::BlockDesc *>("sub_block");
      VLOG(3) << "Conditional Grad block.idx = " << block->ID()
              << ", scope = " << &cur_scope;

      if (FLAGS_control_flow_use_new_executor) {
        LOG_FIRST_N(INFO, 1)
            << "[ControlFlow][ConditionalGradBlock] New Executor is Running.";
        if (!core_ || !platform::is_same_place(core_->GetPlace(), dev_place)) {
          VLOG(10) << "[interpreterCore cache]" << core_.get();
          VLOG_IF(10, core_)
              << platform::is_same_place(core_->GetPlace(), dev_place);
          std::set<std::string> skip_gc_vars(inside_grads.begin(),
                                             inside_grads.end());
          core_.reset(new InterpreterCore(dev_place,
                                          *block,
                                          skip_gc_vars,
                                          &cur_scope,
                                          /* used_for_jit */ false,
                                          /* used_for_control_flow_op */ true));
          VLOG(10) << "[interpreterCore cache]"
                   << "new created:" << core_;
        } else {
          BuildScopeForControlFlowOp(*core_, *block, &cur_scope);
          core_->reset_scope(&cur_scope);
        }
        core_->Run({}, false);

      } else {
        if (!exec_ || !platform::is_same_place(exec_->GetPlace(), dev_place)) {
          auto &pdesc = *block->Program();
          exec_.reset(new Executor(dev_place));
          if (FLAGS_use_mkldnn) exec_->EnableMKLDNN(pdesc);
          ctx_ = exec_->Prepare(pdesc, block->ID(), inside_grads, false);
#ifdef PADDLE_WITH_MKLDNN
          platform::AttachPointerHashToMKLDNNKey(exec_.get(), dev_place);
          platform::RegisterModelLayout(ctx_->ops_, dev_place);
#endif
        }
        exec_->RunPreparedContext(ctx_.get(),
                                  &cur_scope,
                                  /* create_local_scope */ false,
                                  /* create_vars */ true,
                                  /* keep_kids */ true);
      }

      AssignLocalGradientToParentScope(
          dev_place, cur_scope, scope, inside_grads, outside_grads, inputs);
      return;
    }

    AssignZeroToParentScope(dev_place, scope, inputs, outside_grads);
  }

 private:
  mutable std::shared_ptr<Executor> exec_{nullptr};
  mutable std::unique_ptr<ExecutorPrepareContext> ctx_{nullptr};
  mutable std::shared_ptr<InterpreterCore> core_{nullptr};

 private:
  void AssignLocalGradientToParentScope(
      const platform::Place &place,
      const framework::Scope &cur_scope,
      const framework::Scope &parent_scope,
      const std::vector<std::string> &inside_grads,
      const std::vector<std::string> &outside_grads,
      const std::vector<std::string> &inputs) const {
    std::vector<std::string> assign_zero_outside_grads;
    std::vector<std::string> assign_zero_inputs;
    for (size_t i = 0; i < outside_grads.size(); ++i) {
      const std::string &outside_grad_name = outside_grads[i];
      const std::string &inside_grad_name = inside_grads[i];
      VLOG(4) << "[assign local]"
              << "inside_grad_name = " << inside_grad_name
              << ", outside_grad_name = " << outside_grad_name;
      framework::Variable *outside_var =
          parent_scope.FindVar(outside_grad_name);
      if (outside_var == nullptr) {
        continue;
      }
      framework::Variable *inside_var =
          cur_scope.FindLocalVar(inside_grad_name);
      if (inside_var == nullptr) {
        assign_zero_outside_grads.emplace_back(outside_grad_name);
        assign_zero_inputs.emplace_back(inputs[i]);
        continue;
      }
      platform::DeviceContext *dev_ctx =
          platform::DeviceContextPool::Instance().Get(place);
      framework::VisitVarType(*inside_var,
                              AssignFunctor(outside_var, *dev_ctx));
    }
    // Assign zero to the grad_vars that are in outside_grads but not in
    // inside_grads
    AssignZeroToParentScope(
        place, parent_scope, assign_zero_inputs, assign_zero_outside_grads);
  }

  void AssignZeroToParentScope(
      const platform::Place &place,
      const framework::Scope &scope,
      const std::vector<std::string> &inputs,
      const std::vector<std::string> &outside_grads) const {
    for (size_t i = 0; i < outside_grads.size(); ++i) {
      const std::string &outside_grad_name = outside_grads[i];
      const std::string &input_name = inputs[i];
      VLOG(4) << "[assign zero]"
              << "input_name = " << input_name
              << ", outside_grad_name = " << outside_grad_name;
      framework::Variable *input_var = scope.FindVar(input_name);
      if (input_var == nullptr) {
        continue;
      }
      framework::Variable *outside_var = scope.FindVar(outside_grad_name);
      if (outside_var == nullptr) {
        continue;
      }

      if (input_var->IsType<phi::DenseTensor>()) {
        PADDLE_ENFORCE_EQ(
            outside_var->IsType<phi::DenseTensor>(),
            true,
            platform::errors::InvalidArgument(
                "Type of outside_var %s is NOT phi::DenseTensor, which "
                "doesn't match input_var %s.",
                outside_grad_name,
                input_name));
        AssignZeroToOutsideTensor(place,
                                  scope,
                                  input_var->Get<phi::DenseTensor>(),
                                  outside_var->GetMutable<phi::DenseTensor>());
      } else if (input_var->IsType<framework::LoDTensorArray>()) {
        PADDLE_ENFORCE_EQ(outside_var->IsType<framework::LoDTensorArray>(),
                          true,
                          platform::errors::InvalidArgument(
                              "Type of outside_var %s is NOT LoDTensorArray, "
                              "which doesn't match input_var %s.",
                              outside_grad_name,
                              input_name));
        const auto &input_tensors = input_var->Get<framework::LoDTensorArray>();
        auto *outside_tensors =
            outside_var->GetMutable<framework::LoDTensorArray>();
        if (outside_tensors->size() == 0U) {
          outside_tensors->resize(input_tensors.size());
        }
        PADDLE_ENFORCE_EQ(input_tensors.size(),
                          outside_tensors->size(),
                          platform::errors::InvalidArgument(
                              "LoDTensorArray outside_var %s doen't have same "
                              "size as input_var %s.",
                              outside_grad_name,
                              input_name));
        for (size_t j = 0; j < input_tensors.size(); ++j) {
          AssignZeroToOutsideTensor(
              place, scope, input_tensors[j], &((*outside_tensors)[j]));
        }
      } else {
        // TODO(huihuangzheng): add support for SelectedRows
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Conditional block grad op doesn't support non-phi::DenseTensor "
            "output "
            "now."));
      }
    }
  }

  void AssignZeroToOutsideTensor(const platform::Place &place,
                                 const framework::Scope &cur_scope,
                                 const phi::DenseTensor &input_tensor,
                                 phi::DenseTensor *outside_tensor) const {
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

template <class T>
struct FilterNoGradInput {};

template <>
struct FilterNoGradInput<framework::OpDesc> {
  static void filter(const framework::BlockDesc *desc,
                     std::vector<std::string> *vec) {
    auto f = [desc](const std::string &name) -> std::string {
      if (name == framework::kEmptyVarName) {
        // don't drop empty var name, you can use Input(name, true) to drop it.
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
    // NOTE(Aurelius84): VarType of Output is phi::DenseTensor by default. In
    // case of Input is {Tensor, LoDTensorArray}, we need synchronous the
    // Input's VarType into Input@GRAD to avoid generating {Tensor, Tensor} as
    // Input@GRAD.
    auto input_size = ctx->InputSize(ConditionalOp::kInputs);
    auto output_size =
        ctx->OutputSize(framework::GradVarName(ConditionalOp::kInputs));
    PADDLE_ENFORCE_EQ(input_size,
                      output_size,
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

    auto fwd_inputs = this->InputGrad(ConditionalOp::kInputs, false);
    FilterNoGradInput<T>::filter(this->GetForwardOpBlock(), &fwd_inputs);
    grad_op->SetOutput(framework::GradVarName(ConditionalOp::kInputs),
                       fwd_inputs);
    grad_op->SetBlockAttr("sub_block", this->grad_block_[0]);
    grad_op->SetAttr("is_scalar_condition",
                     this->GetAttr("is_scalar_condition"));
  }
};

}  // namespace operators
}  // namespace paddle

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
