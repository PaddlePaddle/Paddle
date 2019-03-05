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

template <typename T>
struct DequantizeFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, framework::Tensor* out) {
    auto in_e = framework::EigenVector<T>::Flatten(*in);
    const T* scale_factor = scale->data<T>();
    auto out_e = framework::EigenVector<T>::Flatten(*out);

    auto& dev = *dev_ctx.eigen_device();
    out_e.device(dev) = (scale_factor[0] / max_range) * in_e;
  }
};

template struct DequantizeFunctor<platform::CPUDeviceContext, float>;
template struct DequantizeFunctor<platform::CPUDeviceContext, double>;

class FakeDequantizeMaxAbsOp : public framework::OperatorWithKernel {
 public:
  FakeDequantizeMaxAbsOp(const std::string& type,
                         const framework::VariableNameMap& inputs,
                         const framework::VariableNameMap& outputs,
                         const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FakeDequantizeMaxAbsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FakeDequantizeMaxAbsOp should not be null.");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class FakeDequantizeMaxAbsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input with float-32/64 type is the "
             "low precision tensor.");
    AddInput("Scale", "(float) The scale in quantization stage.");
    AddOutput("Out",
              "(Tensor) The output is the dequantized high "
              "precision tensor.");
    AddAttr<float>("max_range", "(float) The max range in quantization stage.");
    AddComment(R"DOC(
FakeDequantizeMaxAbsOp operator.

This calculation is an opposite operation of FakeQuantizeMaxAbsOp:

$$Out = \frac{scale*X}{ max_range }$$

)DOC");
  }
};

class FakeChannelWiseDequantizeMaxAbsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of FakeChannelWiseDequantizeMaxAbsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("WeightScales"),
                   "Input(WeightScales) of FakeChannelWiseDequantizeMaxAbsOp "
                   "should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of FakeChannelWiseDequantizeMaxAbsOp should not be null.");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class FakeChannelWiseDequantizeMaxAbsOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input with float-32/64 type is the "
             "low precision tensor.");
    AddInput("ActivationScale",
             "(float) The activation scale in quantization stage.")
        .AsDispensable();
    AddInput("WeightScales",
             "(float array) The weight scales in quantization stage.");
    AddOutput("Out",
              "(Tensor) The output is the dequantized high "
              "precision tensor.");
    AddAttr<int>("activation_bits", "Quantization bit number for activation.")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE(bit_length >= 1 && bit_length <= 16,
                         "'activation_bits' should be between 1 and 16.");
        });
    AddAttr<int>("weight_bits", "Quantization bit number for weights.")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE(bit_length >= 1 && bit_length <= 16,
                         "'weight_bits' should be between 1 and 16.");
        });

    AddComment(R"DOC(
FakeChannelWiseDequantizeMaxAbsOp operator.

This calculation is an opposite operation of FakeChannelWiseQuantizeMaxAbsOp:

$$Out_c = \frac{ActivationScale*WeightScale_c*X_c}{(2^{weight\_bits-1}-1)*(2^{activation\_bits-1}-1)}$$

In the above formula, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$

Notes: Tha per-channel quantization is only applied to weights(channel size scale).
And the activations use per-layer quantization(only one scale).
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

REGISTER_OPERATOR(fake_channel_wise_dequantize_max_abs,
                  ops::FakeChannelWiseDequantizeMaxAbsOp,
                  ops::FakeChannelWiseDequantizeMaxAbsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(fake_channel_wise_dequantize_max_abs,
                       ops::FakeChannelWiseDequantizeMaxAbsKernel<CPU, float>,
                       ops::FakeChannelWiseDequantizeMaxAbsKernel<CPU, double>);
