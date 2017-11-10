/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/elementwise_div_op.h"
#include "paddle/operators/elementwise_op.h"

namespace paddle {
namespace operators {
class ElementwiseDivOpMaker : public ElementwiseOpMaker {
 public:
  ElementwiseDivOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : ElementwiseOpMaker(proto, op_checker) {
    SetComment("Div", "$Out = X / Y$");
    AddComment(comment_);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(elementwise_div, ops::ElementwiseOp, ops::ElementwiseDivOpMaker,
            elementwise_div_grad, ops::ElementwiseOpGrad);
REGISTER_OP_CPU_KERNEL(
    elementwise_div,
    ops::ElementwiseDivKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUPlace, float>);
