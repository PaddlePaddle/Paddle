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
               const phi::Place &dev_place) const override {}

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
