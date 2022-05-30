/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/quantize_linear_op.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/impl/clip_kernel_impl.h"

namespace paddle {
namespace operators {

template <typename T>
struct ChannelDequantizeFunctorV2<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, const int quant_axis, framework::Tensor* out) {
    // Dequant op is before quantized op
    // Dequantize the weight of quantized op
    auto in_dims = in->dims();
    const int64_t channel = in_dims[quant_axis];
    const T* scale_factor = scale->data<T>();
    if (quant_axis == 0) {
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_factor[i];
        framework::Tensor one_channel_in = in->Slice(i, i + 1);
        framework::Tensor one_channel_out = out->Slice(i, i + 1);
        auto in_e = framework::EigenVector<T>::Flatten(one_channel_in);
        auto out_e = framework::EigenVector<T>::Flatten(one_channel_out);
        auto& dev = *dev_ctx.eigen_device();
        out_e.device(dev) = in_e * s / max_range;
      }
    } else if (quant_axis == 1) {
      int64_t out_iter = 1;
      for (int i = 0; i < quant_axis; i++) {
        out_iter *= in_dims[i];
      }
      int64_t step_i = in->numel() / out_iter;
      int64_t step_j = in->numel() / (out_iter * channel);
      auto* in_data = in->data<T>();
      auto* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
      for (int64_t i = 0; i < out_iter; i++) {
        for (int64_t j = 0; j < channel; j++) {
          auto* cur_in = in_data + i * step_i + j * step_j;
          auto* cur_out = out_data + i * step_i + j * step_j;
          T s = scale_factor[j];
          for (int64_t k = 0; k < step_j; k++) {
            *cur_out = (*cur_in) * s / max_range;
            ++cur_in;
            ++cur_out;
          }
        }
      }
    }
  }
};

template struct DequantizeFunctor<platform::CPUDeviceContext, float>;
template struct DequantizeFunctor<platform::CPUDeviceContext, double>;
template struct ChannelDequantizeFunctorV2<platform::CPUDeviceContext, float>;
template struct ChannelDequantizeFunctorV2<platform::CPUDeviceContext, double>;

class QuantizeLinearOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "QuantizeLinear");
    OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "QuantizeLinear");
    OP_INOUT_CHECK(ctx->HasInput("ZeroPoint"), "Input", "ZeroPoint",
                   "QuantizeLinear");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "QuantizeLinear");
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    int quant_axis = ctx->Attrs().Get<int>("quant_axis");
    if (ctx->HasOutput("OutScale")) {
      if (quant_axis < 0) {
        ctx->SetOutputDim("OutScale", {1});
      } else {
        ctx->SetOutputDim("OutScale", {ctx->GetInputDim("X")[quant_axis]});
      }
    }
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class QuantizeLinearOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("Scale", "(Tensor) Input is float data type.");
    AddInput("ZeroPoint", "(Tensor) Input is float data type.");
    AddOutput("Y",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current scale").AsDispensable().AsExtra();
    AddAttr<int>("quant_axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0)
        .AddCustomChecker([](const int& quant_axis) {
          PADDLE_ENFORCE_EQ(
              quant_axis == 0 || quant_axis == 1 || quant_axis == -1, true,
              platform::errors::InvalidArgument(
                  "'quant_axis' should be 0 or 1, but "
                  "the received is %d",
                  quant_axis));
        });
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16, true,
                            platform::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(true);
    AddComment(R"DOC(
The scale of QuantizeLinear operator is a vector.
In detail, each channel of the input X has a scale value.
$$scale_c = max(abs(X_c))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out_c = round(\frac{X_c * range} {scale_c})$$
In above three formulas, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    quantize_linear, ops::QuantizeLinearOp, ops::QuantizeLinearOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(quantize_linear, ops::QuantizeLinearKernel<CPU, float>);

REGISTER_OPERATOR(
    dequantize_linear, ops::QuantizeLinearOp, ops::QuantizeLinearOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(dequantize_linear,
                       ops::DeQuantizeLinearKernel<CPU, float, float>,
                       ops::DeQuantizeLinearKernel<CPU, int8_t, float>,
                       ops::DeQuantizeLinearKernel<CPU, double, double>);
