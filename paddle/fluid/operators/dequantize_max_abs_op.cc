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

#include "paddle/fluid/operators/dequantize_max_abs_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

template <typename T>
struct DequantizeFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  float max_range, framework::Tensor* out) {
    const float* scale_factor = scale->data<float>();
    const T* input_data = in->data<T>();
    float* output_data = out->mutable_data<float>(dev_ctx.GetPlace());
    auto input_dims = in->dims();
    int ind = 1;
    for (size_t i = 0; i < (unsigned)input_dims.size(); i++) {
      ind *= input_dims[i];
    }
    for (size_t i = 0; i < (unsigned)ind; i++) {
      output_data[i] = (scale_factor[0] / max_range) * input_data[i];
    }
  }
};

template struct DequantizeFunctor<platform::CPUDeviceContext, int8_t>;

class DequantizeMaxAbsOp2 : public framework::OperatorWithKernel {
 public:
  DequantizeMaxAbsOp2(const std::string& type,
                      const framework::VariableNameMap& inputs,
                      const framework::VariableNameMap& outputs,
                      const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of DequantizeMaxAbsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of DequantizeMaxAbsOp should not be null.");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto type = framework::OpKernelType(
        ctx.Input<framework::LoDTensor>("X")->type(), ctx.device_context());
    return type;
  }
};

class DequantizeMaxAbsOpMaker2 : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(int8 Tensor) The input with int8 type is the "
             "low precision tensor.");
    AddInput("Scale", "(float) The scale in quantization stage.");
    AddOutput("Out",
              "(float32 Tensor) The output is the dequantized high "
              "precision tensor.");
    AddAttr<float>("max_range", "(float) The max range in quantization stage.");
    AddComment(R"DOC(
DequantizeMaxAbsOp operator.

This calculation is an opposite operation of QuantizeMaxAbsOp:

$$Out = \frac{scale*X}{ max_range }$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(dequantize_max_abs, ops::DequantizeMaxAbsOp2,
                  ops::DequantizeMaxAbsOpMaker2,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(dequantize_max_abs,
                       ops::DequantizeMaxAbsKernel2<CPU, int8_t>);
