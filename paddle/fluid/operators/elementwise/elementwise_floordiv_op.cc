/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/elementwise/elementwise_floordiv_op.h"

#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
class ElementwiseFloorDivOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "FloorDiv"; }
  std::string GetEquation() const override { return "Out = X // Y"; }

  void AddInputX() override {
    AddInput("X",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64.");
  }

  void AddInputY() override {
    AddInput("Y",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64.");
  }

  std::string GetOpFuntionality() const override {
    return "Floor divide two tensors element-wise";
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(elementwise_floordiv, ops::ElementwiseOp,
                             ops::ElementwiseFloorDivOpMaker);

REGISTER_OP_CPU_KERNEL(
    elementwise_floordiv,
    ops::ElementwiseFloorDivKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseFloorDivKernel<paddle::platform::CPUDeviceContext,
                                   int64_t>);

REGISTER_OP_VERSION(elementwise_floordiv)
    .AddCheckpoint(
        R"ROC(Register elementwise_floordiv for adding the attribute of Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_floordiv.",
            1.0f));
