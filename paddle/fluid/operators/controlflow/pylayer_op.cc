// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/controlflow/pylayer_op.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/fluid/operators/controlflow/control_flow_op_helper.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle::operators {

namespace {  // NOLINT
enum class PyLayerBlockIndex { kFORWARD = 0, kBACKWARD = 1, kNONE = 2 };
}  // namespace

const char PyLayerOp::kInputs[] = "Input";        // NOLINT
const char PyLayerOp::kOutputs[] = "Out";         // NOLINT
const char PyLayerOp::kScope[] = "Scope";         // NOLINT
const char PyLayerOp::kSkipEagerDeletionVars[] =  // NOLINT
    "skip_eager_deletion_vars";
const char PyLayerOp::kBlocks[] = "blocks";  // NOLINT

void PyLayerOp::CreateInterpreter(
    const phi::Place &dev_place,
    const framework::BlockDesc &block,
    framework::Scope *cur_scope,
    const std::vector<std::string> &skip_vars) const {
  if (!core_ || !phi::is_same_place(core_->GetPlace(), dev_place)) {
    VLOG(10) << "[interpreterCore cache]" << core_.get();
    VLOG_IF(10, core_) << phi::is_same_place(core_->GetPlace(), dev_place);

    framework::interpreter::ExecutionConfig execution_config;
    execution_config.create_local_scope = false;
    execution_config.used_for_control_flow_op = true;
    execution_config.skip_gc_vars =
        std::set<std::string>(skip_vars.begin(), skip_vars.end());

    core_.reset(new framework::InterpreterCore(
        dev_place, block, cur_scope, execution_config));
    VLOG(10) << "[interpreterCore] created:" << core_;
  } else {
    BuildScopeForControlFlowOp(*core_, block, cur_scope);
    core_->reset_scope(cur_scope);
  }
}

class PyLayerForwardOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(PyLayerOp::kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(PyLayerOp::kOutputs, "The output variables of the sub-block.")
        .AsDuplicable();
    AddOutput(
        PyLayerOp::kScope,
        "(std::vector<Scope*>) The scope of static pylayer block, used for "
        "passing intermediate variables between forward and backward.");
    AddAttr<std::vector<framework::BlockDesc *>>(
        "blocks",
        "The blocks of PyLayer operator where blocks[0] indicates the forward "
        "block and blocks[1] indicates the backward block.");
    AddComment(R"DOC(PyLayer operator

The PyLayer Operator is designed to support `@to_static` for `PyLayer in Dynamic Graph`.


)DOC");
  }
};

class PyLayerForwardOp : public PyLayerOp {
 public:
  PyLayerForwardOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : PyLayerOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const phi::Place &dev_place) const override {
    auto *scope_var = scope.FindVar(Output(kScope));
    PADDLE_ENFORCE_NOT_NULL(
        scope_var,
        common::errors::PreconditionNotMet(
            "Expect Scope variable to be set in pylayer_op, but "
            "got a null Scope variable. Please set the Scope variable."));

    auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
    scopes->resize(1);
    scopes->front() = &scope.NewScope();

    auto &cur_scope = *scopes->front();
    auto &blocks =
        Attr<std::vector<framework::BlockDesc *>>(PyLayerOp::kBlocks);
    PADDLE_ENFORCE_GT(
        blocks.size(),
        0,
        common::errors::InvalidArgument(
            "Expect blocks contains at least 1 block, but got: %d",
            blocks.size()));

    framework::BlockDesc *forward_block =
        blocks[static_cast<size_t>(PyLayerBlockIndex::kFORWARD)];
    VLOG(3) << "PyLayer forward_block block.idx = " << forward_block->ID()
            << ", scope = " << &cur_scope;

    auto &skip_vars = Attr<std::vector<std::string>>(kSkipEagerDeletionVars);

    LOG_FIRST_N(INFO, 1) << "[ControlFlow][PyLayer] New Executor is Running.";

    CreateInterpreter(dev_place, *forward_block, &cur_scope, skip_vars);
    PADDLE_ENFORCE_NOT_NULL(core_, common::errors::Fatal("core_ is nullptr"));
    core_->Run({}, false);
  }
};

class PyLayerForwardInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    // NOTE(MarioLulab): do nothing.
  }
};

template <typename T>
class PyLayerBackwardMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("pylayer_grad");
    grad_op->SetInput(PyLayerOp::kInputs, this->Input(PyLayerOp::kInputs));
    grad_op->SetInput(framework::GradVarName(PyLayerOp::kOutputs),
                      this->OutputGrad(PyLayerOp::kOutputs));
    grad_op->SetInput(PyLayerOp::kScope, this->Output(PyLayerOp::kScope));

    auto fwd_inputs = this->InputGrad(PyLayerOp::kInputs, false);
    grad_op->SetOutput(framework::GradVarName(PyLayerOp::kInputs), fwd_inputs);

    const std::vector<framework::BlockDesc *> &blocks = PADDLE_GET_CONST(
        std::vector<framework::BlockDesc *>, this->GetAttr(PyLayerOp::kBlocks));
    PADDLE_ENFORCE_GT(
        blocks.size(),
        static_cast<size_t>(PyLayerBlockIndex::kBACKWARD),
        common::errors::InvalidArgument(
            "Expect blocks contains at least 2 block, but got: %d",
            blocks.size()));
    grad_op->SetBlockAttr(
        "backward_block",
        blocks[static_cast<size_t>(PyLayerBlockIndex::kBACKWARD)]);
  }
};

class PyLayerBackwardOp : public PyLayerOp {
 public:
  PyLayerBackwardOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : PyLayerOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const phi::Place &dev_place) const override {
    const auto &inputs = Inputs(PyLayerOp::kInputs);
    const auto &outside_grads =
        Outputs(framework::GradVarName(PyLayerOp::kInputs));
    std::vector<std::string> inside_grads;
    inside_grads.reserve(inputs.size());
    for (auto &in : inputs) {
      inside_grads.emplace_back(framework::GradVarName(in));
    }

    PADDLE_ENFORCE_EQ(
        inside_grads.size(),
        outside_grads.size(),
        common::errors::InvalidArgument(
            "Mismatch inside_grads.size(): %d, and outside_grads.size(): %d",
            inside_grads.size(),
            outside_grads.size()));

    auto *scope_var = scope.FindVar(Input(PyLayerOp::kScope));
    PADDLE_ENFORCE_NOT_NULL(
        scope_var,
        common::errors::PreconditionNotMet(
            "Expect Scope variable to be set in pylayer_op, but "
            "got a null Scope variable. Please set the Scope variable."));
    auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
    PADDLE_ENFORCE_GT(
        scopes.size(),
        0,
        common::errors::InvalidArgument(
            "Expect Scope variable contains at least 1 scope, but got: %d",
            scopes.size()));
    framework::Scope &cur_scope = *(scopes[0]);

    auto *backward_block = Attr<framework::BlockDesc *>("backward_block");
    VLOG(3) << "Static PyLayer backward block.idx = " << backward_block->ID()
            << ", scope = " << &cur_scope;

    LOG_FIRST_N(INFO, 1)
        << "[ControlFlow][PyLayerBackwardOp] New Executor is Running.";

    CreateInterpreter(dev_place, *backward_block, &cur_scope, inside_grads);
    PADDLE_ENFORCE_NOT_NULL(core_, common::errors::Fatal("core_ is nullptr"));

    core_->Run({}, false);

    // NOTE: It's necessary. The reason of associating `inside_grads` and
    // `outside_grads` at runtime `RunImpl` instead of `assign` op at block is
    // that the Var name of grad_op's outputs may be changed in the
    // `append_backward` function (e.g. `_addup_repetitive_outputs_`).
    AssignLocalGradientToParentScope(
        dev_place, cur_scope, scope, inside_grads, outside_grads, inputs);

    // Release the cur_scope, otherwise memory leakage occurs.
    scope.DeleteScope(&cur_scope);
    return;
  }
};

class PyLayerBackwardInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    if (context->HasInputs(PyLayerOp::kInputs) &&
        context->HasOutputs(framework::GradVarName(PyLayerOp::kInputs))) {
      context->SetOutputsDim(framework::GradVarName(PyLayerOp::kInputs),
                             context->GetInputsDim(PyLayerOp::kInputs));
    }
  }
};

class PyLayerBackwardInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto forward_input_size = ctx->InputSize(PyLayerOp::kInputs);
    auto backward_output_size =
        ctx->OutputSize(framework::GradVarName(PyLayerOp::kInputs));
    PADDLE_ENFORCE_EQ(forward_input_size,
                      backward_output_size,
                      common::errors::InvalidArgument(
                          "input_size and output_size should be equal for "
                          "pylayer_grad op."));
    for (size_t i = 0; i < backward_output_size; ++i) {
      ctx->SyncTypeAndDataType(PyLayerOp::kInputs,
                               framework::GradVarName(PyLayerOp::kInputs),
                               static_cast<int>(i));
    }
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;
REGISTER_OPERATOR(pylayer,
                  ops::PyLayerForwardOp,
                  ops::PyLayerForwardInferShape,
                  ops::PyLayerForwardOpProtoMaker,
                  ops::PyLayerBackwardMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(pylayer_grad,
                  ops::PyLayerBackwardOp,
                  ops::PyLayerBackwardInferShape,
                  ops::PyLayerBackwardInferVarType);
