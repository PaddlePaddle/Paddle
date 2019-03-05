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
#include <vector>

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
    PADDLE_ENFORCE(ctx->HasInputs("Scales"),
                   "Input(Scales) of FakeChannelWiseDequantizeMaxAbsOp "
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
    AddInput("Scales",
             "(Tensors) The scales in quantization stage. "
             "Now, `Scales` is a vector with at most two tensors. "
             "If Scales has two elements, the second tensor should only have "
             "one value.")
        .AsDuplicable();
    AddOutput("Out",
              "(Tensor) The output is the dequantized high "
              "precision tensor.");
    AddAttr<std::vector<int>>(
        "quant_bits",
        "Quantization bit numbers in quantization stage. "
        "The size of `quant_bits` should be equal to the size of `Scales`.")
        .SetDefault({8});

    AddComment(R"DOC(
FakeChannelWiseDequantizeMaxAbsOp operator.

This calculation is an opposite operation of FakeChannelWiseQuantizeMaxAbsOp:

$$Out_c = \frac{X_c\prod_{i=1}^{n}Scales_{ic}}{\prod_{i=1}^{n}(2^{quant\_bits_i-1}-1)}$$

In the above formula, the range value of $c$ can be represented as $0 \leq c \lt \ the\ channel\ number\ of\ X$.
Besides, the size of $quant\_bits$ should be equal to the size of $Scales$, and it is called $n$  in the formula.

Notes: In general, the per-channel quantization is only applied to weights and the activations use per-layer quantization.
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
