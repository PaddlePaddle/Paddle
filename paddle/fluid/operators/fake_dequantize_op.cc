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

#include "paddle/fluid/operators/fake_dequantize_op.h"
#include <string>

namespace paddle {
namespace operators {

class FakeDequantizeMaxAbsOp : public framework::OperatorWithKernel {
 public:
  FakeDequantizeMaxAbsOp(const std::string &type,
                         const framework::VariableNameMap &inputs,
                         const framework::VariableNameMap &outputs,
                         const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FakeDequantizeMaxAbsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FakeDequantizeMaxAbsOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class FakeDequantizeMaxAbsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input with float-32/64 type is the "
             "low precision tensor.");
    AddOutput("Out",
              "(Tensor) The output is the dequantized high "
              "precision tensor.");
    AddAttr<int>("num_bits",
                 "(int) `num_bits` is the quantization level bits, "
                 "such as 2, 5, 8.");
    AddAttr<float>("scale",
                   "(float) The maximum absolute value of low precision tensor."
                   "It is usually calculated by the fake_quantize_max_abs_op.");
    AddComment(R"DOC(
FakeDequantizeMaxAbsOp operator.

This calculation is an opposite operation of FakeQuantizeMaxAbsOp:

$$Out = \frac{scale*X}{2^{num_bits} - 1}$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(fake_dequantize_max_abs, ops::FakeDequantizeMaxAbsOp,
                  ops::FakeDequantizeMaxAbsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(fake_dequantize_max_abs,
                       ops::FakeDequantizeMaxAbsKernel<CPU, float>,
                       ops::FakeDequantizeMaxAbsKernel<CPU, double>);
