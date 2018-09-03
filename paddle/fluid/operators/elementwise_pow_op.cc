/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/elementwise_pow_op.h"
#include <string>
#include "paddle/fluid/operators/elementwise_op.h"

namespace paddle {
namespace operators {
class ElementwisePowOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Pow"; }
  std::string GetEquation() const override { return "Out = X ^ Y"; }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(elementwise_pow, ops::ElementwiseOp,
                             ops::ElementwisePowOpMaker);
REGISTER_OP_CPU_KERNEL(
    elementwise_pow,
    ops::ElementwisePowKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwisePowKernel<paddle::platform::CPUDeviceContext, double>);
