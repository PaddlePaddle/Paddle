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

#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_mul_one_dnn_op.h"
#include "paddle/fluid/platform/complex.h"

namespace paddle {
namespace operators {


class ElementwiseMulOneDNNOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Mul"; }
  std::string GetEquation() const override { return "Out = X \\\\odot Y"; }

  void AddInputX() override {
    AddInput("X",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  void AddInputY() override {
    AddInput("Y",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  std::string GetOpFuntionality() const override {
    return "Multiply two tensors element-wise";
  }

  void Make() override final {
    ElementwiseOpMaker::Make();
    
    /* activation parameters */
    AddAttr<std::string>("activation_type",
                         "Activation type used in elementwise operator.")
        .SetDefault("")
        .AsExtra();
    AddAttr<float>(
        "activation_alpha",
        "Activation alpha parameter type used in elementwise operator.")
        .SetDefault(0.0f)
        .AsExtra();
    AddAttr<float>(
        "activation_beta",
        "Activation beta parameter type used in elementwise operator.")
        .SetDefault(0.0f)
        .AsExtra();
    AddAttr<float>(
        "activation_scale",
        "Activation scale parameter type used in elementwise operator.")
        .SetDefault(1.0f)
        .AsExtra();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(elementwise_mul_one_dnn, ops::ElementwiseMulOneDNNOp,
                  ops::ElementwiseMulOneDNNOpMaker, ops::ElementwiseOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    elementwise_mul_one_dnn,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<float>>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<double>>);

REGISTER_OP_VERSION(elementwise_mul_one_dnn)
    .AddCheckpoint(
        R"ROC(Register elementwise_mul_one_dnn for adding the attribute of Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_mul_one_dnn.",
            1.0f));
