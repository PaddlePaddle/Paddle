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

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif
#include "paddle/common/flags.h"

COMMON_DECLARE_bool(use_mkldnn);
namespace paddle {
namespace framework {
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

/* We will implement the op with block separately in the future.
 * The main reason is that some of the training requirements
 * in these OPS can lead to problems(such as memory leaks) during inference.
 */
class ConditionalBlockInferOp : public ConditionalOp {
 public:
  ConditionalBlockInferOp(const std::string &type,
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
      auto xs = InputTensors(scope, "Cond");
      need_run = ScalarCondition(xs);
    } else {
      // When is_scalar_condition is False, the conditional variable maybe a
      // vector or tensor, whether need to execute the operators in sub-block
      // depends on the input variables (Input).
      auto xs = InputTensors(scope, "Input");
      need_run =
          std::all_of(xs.begin(), xs.end(), [](const phi::DenseTensor *t) {
            return t->numel() != 0;
          });
    }

    if (need_run) {
      auto *scope_var = scope.FindVar(Output("Scope"));
      PADDLE_ENFORCE_NOT_NULL(
          scope_var,
          common::errors::PreconditionNotMet(
              "Scope must be set in ConditionalBlockInferOp."));
      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
      auto &cur_scope = *scopes->front();

      auto *block = Attr<framework::BlockDesc *>("sub_block");
      VLOG(3) << "Conditional block.idx = " << block->ID()
              << ", scope = " << &cur_scope;

      if (!exec_ || !phi::is_same_place(exec_->GetPlace(), dev_place)) {
        auto &pdesc = *block->Program();
        exec_.reset(new framework::Executor(dev_place));
#ifdef PADDLE_WITH_DNNL
        if (FLAGS_use_mkldnn) exec_->EnableMKLDNN(pdesc);
#endif
        ctx_ = exec_->Prepare(
            pdesc, block->ID(), std::vector<std::string>(), false);
#ifdef PADDLE_WITH_DNNL
        if (FLAGS_use_mkldnn) {
          platform::AttachPointerHashToMKLDNNKey(exec_.get(), dev_place);
          platform::RegisterModelLayout(ctx_->ops_, dev_place);
        }
#endif
      }
      exec_->RunPreparedContext(ctx_.get(), &cur_scope, false, true, false);
      scope.DeleteScope(scopes->front());
    }
  }

 private:
  mutable std::shared_ptr<framework::Executor> exec_{nullptr};
  mutable std::unique_ptr<framework::ExecutorPrepareContext> ctx_{nullptr};
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    conditional_block_infer,
    ops::ConditionalBlockInferOp,
    ops::ConditionalBlockOpProtoMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
