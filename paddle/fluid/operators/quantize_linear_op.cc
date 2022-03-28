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
#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
struct Compare {
 public:
  bool operator()(const T a, const T b) { return (std::abs(a) < std::abs(b)); }
};

template <typename T>
struct FindAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const T* in,
                  const int num, T* out) {
    *out = std::abs(*(std::max_element(in + 0, in + num, Compare<T>())));
  }
};

template struct FindAbsMaxFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct FindChannelAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in_tensor, const int quant_axis,
                  T* out_abs_max) {
    // At present, channelwise quantization supports conv2d, depthwise_conv2d
    // conv2d_transpose and mul
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));
    auto* in_data = in_tensor.data<T>();
    auto in_dims = in_tensor.dims();
    const int64_t channel = in_dims[quant_axis];
    if (quant_axis == 0) {
      const int64_t channel_size = in_tensor.numel() / channel;
      for (int64_t i = 0; i < channel; i++) {
        auto* start = in_data + i * channel_size;
        auto* end = in_data + (i + 1) * channel_size;
        out_abs_max[i] =
            std::abs(*(std::max_element(start, end, Compare<T>())));
      }
    } else if (quant_axis == 1) {
      for (int64_t i = 0; i < channel; i++) {
        out_abs_max[i] = 0;
      }
      const int64_t step_i = in_tensor.numel() / in_dims[0];
      const int64_t step_j = in_tensor.numel() / (in_dims[0] * in_dims[1]);
      for (int64_t i = 0; i < in_dims[0]; i++) {
        for (int64_t j = 0; j < in_dims[1]; j++) {
          auto* start = in_data + i * step_i + j * step_j;
          auto* end = in_data + i * step_i + (j + 1) * step_j;
          T abs_max = std::abs(*(std::max_element(start, end, Compare<T>())));
          out_abs_max[j] = std::max(out_abs_max[j], abs_max);
        }
      }
    }
  }
};

template struct FindChannelAbsMaxFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct ClipAndFakeQuantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    T s = scale.data<T>()[0];
    T inv_s = inverse(s);
    platform::Transform<platform::CPUDeviceContext> trans;
    trans(ctx, in.data<T>(), in.data<T>() + in.numel(),
          out->mutable_data<T>(ctx.GetPlace()), ClipFunctor<T>(-s, s));
    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) = (bin_cnt * inv_s * out_e).round();
  }
};

template struct ClipAndFakeQuantFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct ChannelClipAndFakeQuantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, const int quant_axis,
                  framework::Tensor* out) {
    // At present, channelwise quantization supports conv2d, depthwise_conv2d
    // conv2d_transpose and mul
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));
    auto* scale_data = scale.data<T>();
    auto* in_data = in.data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());
    auto in_dims = in.dims();
    const int64_t channel = in_dims[quant_axis];
    platform::Transform<platform::CPUDeviceContext> trans;
    if (quant_axis == 0) {
      const int64_t channel_size = in.numel() / channel;
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_data[i];
        auto* start = in_data + i * channel_size;
        auto* end = in_data + (i + 1) * channel_size;
        trans(ctx, start, end, out_data + i * channel_size,
              ClipFunctor<T>(-s, s));
      }
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_data[i];
        T inv_s = inverse(s);
        framework::Tensor one_channel_out = out->Slice(i, i + 1);
        auto out_e = framework::EigenVector<T>::Flatten(one_channel_out);
        out_e.device(*ctx.eigen_device()) = (bin_cnt * inv_s * out_e).round();
      }
    } else if (quant_axis == 1) {
      const int64_t step_i = in.numel() / in_dims[0];
      const int64_t step_j = in.numel() / (in_dims[0] * in_dims[1]);
      for (int i = 0; i < in_dims[0]; i++) {
        for (int j = 0; j < in_dims[1]; j++) {
          T s = scale_data[j];
          T inv_s = inverse(s);
          auto* start = in_data + i * step_i + j * step_j;
          auto* end = in_data + i * step_i + (j + 1) * step_j;
          auto* cur_out_data = out_data + i * step_i + j * step_j;
          trans(ctx, start, end, cur_out_data, ClipFunctor<T>(-s, s));
          for (int k = 0; k < step_j; k++) {
            cur_out_data[k] = std::round(bin_cnt * inv_s * cur_out_data[k]);
          }
        }
      }
    }
  }
};

template struct ChannelClipAndFakeQuantFunctor<platform::CPUDeviceContext,
                                               float>;

template <typename T>
struct DequantizeFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, framework::Tensor* out) {
    auto in_e = framework::EigenVector<T>::Flatten(*in);
    const T* scale_factor = scale->data<T>();
    auto out_e = framework::EigenVector<T>::Flatten(*out);

    auto& dev = *dev_ctx.eigen_device();
    out_e.device(dev) = in_e * scale_factor[0] / max_range;
  }
};

template <typename T>
struct ChannelDequantizeFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  const int scale_num, T max_range, const int quant_axis,
                  const int x_num_col_dims, framework::Tensor* out) {
    if (scale_num == 1) {
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
  }
};

template struct DequantizeFunctor<platform::CPUDeviceContext, float>;
template struct DequantizeFunctor<platform::CPUDeviceContext, double>;
template struct ChannelDequantizeFunctor<platform::CPUDeviceContext, float>;
template struct ChannelDequantizeFunctor<platform::CPUDeviceContext, double>;

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
