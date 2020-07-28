/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/common_cwise_functors.h"
#include "paddle/fluid/operators/common_cwise_ops.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

// This file almostly contains all the coefficient-wise Operator class and
// OpKernel class

namespace paddle {
namespace operators {
class CwisePowOpMaker : public ElementwiseOpMaker {
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
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace functors = paddle::operators::functors;

REGISTER_OPERATOR(cwise_mul, ops::BinaryOp, ops::CwisePowOpMaker);

// REGISTER_OPERATOR(elementwise_pow_grad, ops::ElementwiseOpGrad);

REGISTER_OP_CPU_KERNEL(cwise_mul,
                       ops::BinaryOpKernel<paddle::platform::CPUDeviceContext,
                                           functors::Mul<float>, float>,
                       ops::BinaryOpKernel<paddle::platform::CPUDeviceContext,
                                           functors::Mul<double>, double>,
                       ops::BinaryOpKernel<paddle::platform::CPUDeviceContext,
                                           functors::Mul<int>, int>,
                       ops::BinaryOpKernel<paddle::platform::CPUDeviceContext,
                                           functors::Mul<int64_t>, int64_t>);
