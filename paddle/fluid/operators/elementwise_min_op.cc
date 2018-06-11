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

#include "paddle/fluid/operators/elementwise_min_op.h"
#include "paddle/fluid/operators/elementwise_op.h"

namespace paddle {
namespace operators {
class ElementwiseMinOpMaker : public ElementwiseOpMaker {
 public:
  ElementwiseMinOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : ElementwiseOpMaker(proto, op_checker) {
    SetComment("Max", "Out = min(X, Y)");
    AddComment(comment_);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(elementwise_min, ops::ElementwiseOp,
                  ops::ElementwiseMinOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(elementwise_min_grad, ops::ElementwiseOpGrad);
REGISTER_OP_CPU_KERNEL(
    elementwise_min,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_min_grad,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
