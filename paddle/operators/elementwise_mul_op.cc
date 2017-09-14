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

#include "paddle/operators/elementwise_op.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class ElementWiseMulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElementWiseCompute<EigenMulFunctor, Place, T>(ctx);
  }
};

template <typename Place, typename T>
class ElementWiseMulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(elementwise_mul, ops::ElementWiseOp, ops::ElementWiseOpMaker,
            elementwise_mul_grad, ops::ElementWiseOpGrad);
REGISTER_OP_CPU_KERNEL(
    elementwise_mul,
    ops::ElementWiseMulKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementWiseMulGradKernel<paddle::platform::CPUPlace, float>);
