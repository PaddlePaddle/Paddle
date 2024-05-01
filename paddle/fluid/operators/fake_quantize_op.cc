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

#include "paddle/fluid/operators/fake_quantize_op.h"

#include <algorithm>
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/clip_kernel_impl.h"

namespace paddle {
namespace operators {

template <typename T>
struct Compare {
 public:
  bool operator()(const T a, const T b) { return (std::abs(a) < std::abs(b)); }
};

template <typename T>
struct FindChannelAbsMaxFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext &ctx,
                  const phi::DenseTensor &in_tensor,
                  const int quant_axis,
                  T *out_abs_max) {
    // At present, channelwise quantization supports conv2d, depthwise_conv2d
    // conv2d_transpose and mul
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1,
        true,
        phi::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                     "the received is %d",
                                     quant_axis));
    auto *in_data = in_tensor.data<T>();
    auto in_dims = in_tensor.dims();
    const int64_t channel = in_dims[quant_axis];
    if (quant_axis == 0) {
      const int64_t channel_size = in_tensor.numel() / channel;
      for (int64_t i = 0; i < channel; i++) {
        auto *start = in_data + i * channel_size;
        auto *end = in_data + (i + 1) * channel_size;
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
          auto *start = in_data + i * step_i + j * step_j;
          auto *end = in_data + i * step_i + (j + 1) * step_j;
          T abs_max = std::abs(*(std::max_element(start, end, Compare<T>())));
          out_abs_max[j] = std::max(out_abs_max[j], abs_max);
        }
      }
    }
  }
};

template struct FindChannelAbsMaxFunctor<phi::CPUContext, float>;

template <typename T>
struct ChannelClipAndFakeQuantFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext &ctx,
                  const phi::DenseTensor &in,
                  const phi::DenseTensor &scale,
                  const int bin_cnt,
                  const int round_type,
                  const int quant_axis,
                  phi::DenseTensor *out) {
    // At present, channelwise quantization supports conv2d, depthwise_conv2d
    // conv2d_transpose and mul
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1,
        true,
        phi::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                     "the received is %d",
                                     quant_axis));
    auto *scale_data = scale.data<T>();
    auto *in_data = in.data<T>();
    auto *out_data = out->mutable_data<T>(ctx.GetPlace());
    auto in_dims = in.dims();
    const int64_t channel = in_dims[quant_axis];
    phi::Transform<phi::CPUContext> trans;
    if (quant_axis == 0) {
      const int64_t channel_size = in.numel() / channel;
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_data[i];
        auto *start = in_data + i * channel_size;
        auto *end = in_data + (i + 1) * channel_size;
        T inv_s = phi::funcs::inverse(s);
        if (round_type == 0) {
          trans(ctx,
                start,
                end,
                out_data + i * channel_size,
                phi::funcs::QuantTensorFunctor<T>(static_cast<T>(bin_cnt),
                                                  inv_s));
        } else {
          trans(ctx,
                start,
                end,
                out_data + i * channel_size,
                phi::ClipFunctor<T>(-s, s));
        }
      }
      if (round_type == 1) {
        for (int64_t i = 0; i < channel; i++) {
          T s = scale_data[i];
          T inv_s = phi::funcs::inverse(s);
          phi::DenseTensor one_channel_out = out->Slice(i, i + 1);
          auto out_e = phi::EigenVector<T>::Flatten(one_channel_out);
          out_e.device(*ctx.eigen_device()) = (bin_cnt * inv_s * out_e).round();
        }
      }
    } else if (quant_axis == 1) {
      const int64_t step_i = in.numel() / in_dims[0];
      const int64_t step_j = in.numel() / (in_dims[0] * in_dims[1]);
      for (int i = 0; i < in_dims[0]; i++) {
        for (int j = 0; j < in_dims[1]; j++) {
          T s = scale_data[j];
          T inv_s = phi::funcs::inverse(s);
          auto *start = in_data + i * step_i + j * step_j;
          auto *end = in_data + i * step_i + (j + 1) * step_j;
          auto *cur_out_data = out_data + i * step_i + j * step_j;
          if (round_type == 0) {
            trans(ctx,
                  start,
                  end,
                  cur_out_data,
                  phi::funcs::QuantTensorFunctor<T>(static_cast<T>(bin_cnt),
                                                    inv_s));
          } else {
            trans(ctx, start, end, cur_out_data, phi::ClipFunctor<T>(-s, s));
            for (int k = 0; k < step_j; k++) {
              cur_out_data[k] = std::round(bin_cnt * inv_s * cur_out_data[k]);
            }
          }
        }
      }
    }
  }
};

template struct ChannelClipAndFakeQuantFunctor<phi::CPUContext, float>;
template <typename T>
struct ChannelClipFakeQuantDequantFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext &ctx,
                  const phi::DenseTensor &in,
                  const phi::DenseTensor &scale,
                  const int bin_cnt,
                  const int round_type,
                  const int quant_axis,
                  phi::DenseTensor *out) {
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1,
        true,
        phi::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                     "the received is %d",
                                     quant_axis));

    auto *scale_data = scale.data<T>();
    auto *in_data = in.data<T>();
    auto *out_data = out->mutable_data<T>(ctx.GetPlace());
    auto in_dims = in.dims();
    const int64_t channel = in_dims[quant_axis];
    phi::Transform<phi::CPUContext> trans;
    if (quant_axis == 0) {
      const int64_t channel_size = in.numel() / channel;
      for (int i = 0; i < channel; i++) {
        T s = scale_data[i];
        auto *start = in_data + i * channel_size;
        auto *end = in_data + (i + 1) * channel_size;
        if (round_type == 0) {
          T inv_s = phi::funcs::inverse(s);
          trans(ctx,
                start,
                end,
                out_data + i * channel_size,
                phi::funcs::QuantTensorFunctor<T>(static_cast<T>(bin_cnt),
                                                  inv_s));
        } else {
          trans(ctx,
                start,
                end,
                out_data + i * channel_size,
                phi::ClipFunctor<T>(-s, s));
        }
      }
      for (int i = 0; i < channel; i++) {
        T s = scale_data[i];
        phi::DenseTensor one_channel_out = out->Slice(i, i + 1);
        auto out_e = phi::EigenVector<T>::Flatten(one_channel_out);
        if (round_type == 0) {
          out_e.device(*ctx.eigen_device()) =
              out_e * s / static_cast<T>(bin_cnt);
        } else {
          T inv_s = phi::funcs::inverse(s);
          out_e.device(*ctx.eigen_device()) =
              (bin_cnt * inv_s * out_e).round() * s / static_cast<T>(bin_cnt);
        }
      }
    } else if (quant_axis == 1) {
      const int64_t step_i = in.numel() / in_dims[0];
      const int64_t step_j = in.numel() / (in_dims[0] * in_dims[1]);
      for (int i = 0; i < in_dims[0]; i++) {
        for (int j = 0; j < in_dims[1]; j++) {
          T s = scale_data[j];
          T inv_s = phi::funcs::inverse(s);
          auto *start = in_data + i * step_i + j * step_j;
          auto *end = in_data + i * step_i + (j + 1) * step_j;
          auto *cur_out_data = out_data + i * step_i + j * step_j;
          if (round_type == 0) {
            trans(ctx,
                  start,
                  end,
                  cur_out_data,
                  phi::funcs::QuantTensorFunctor<T>(static_cast<T>(bin_cnt),
                                                    inv_s));
          } else {
            trans(ctx, start, end, cur_out_data, phi::ClipFunctor<T>(-s, s));
          }
          for (int k = 0; k < step_j; k++) {
            if (round_type == 0) {
              cur_out_data[k] = cur_out_data[k] * s / static_cast<T>(bin_cnt);
            } else {
              cur_out_data[k] = std::round(bin_cnt * inv_s * cur_out_data[k]) *
                                s / static_cast<T>(bin_cnt);
            }
          }
        }
      }
    }
  }
};

class FakeQuantOrWithDequantAbsMaxOp : public framework::OperatorWithKernel {
 public:
  FakeQuantOrWithDequantAbsMaxOp(const std::string &type,
                                 const framework::VariableNameMap &inputs,
                                 const framework::VariableNameMap &outputs,
                                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "FakeQuantOrWithDequantAbsMaxOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FakeQuantOrWithDequantAbsMaxOp");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"),
                   "Output",
                   "OutScale",
                   "FakeQuantOrWithDequantAbsMaxOp");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }
};

class FakeQuantOrWithDequantAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current scale");
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int &bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                            true,
                            phi::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddComment(R"DOC(
This is a Base Op which supports FakeQuantAbsMaxOpMaker and FakeQuantDequantAbsMaxOpMaker.
FakeQuantAbsMaxOp operator is used in the dynamic quantization.

$$scale = max(abs(X))$$
$$range = 2^{bit_length - 1} - 1$$
$$Out = round(X/scale * range)$$

FakeQuantDequantAbsMaxOp operator does the abs_max quantization and then dequantization.

$$scale = max(abs(X))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out = round(X/scale * range) * scale / range$$

)DOC");
  }
};

class FakeChannelWiseQuantizeAbsMaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "FakeChannelWiseQuantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FakeChannelWiseQuantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"),
                   "Output",
                   "OutScale",
                   "FakeChannelWiseQuantizeAbsMax");
    int quant_axis = ctx->Attrs().Get<int>("quant_axis");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {ctx->GetInputDim("X")[quant_axis]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class FakeChannelWiseQuantizeAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current channel wise scale");
    AddAttr<int>("quant_axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0)
        .AddCustomChecker([](const int &quant_axis) {
          PADDLE_ENFORCE_EQ(
              quant_axis == 0 || quant_axis == 1,
              true,
              phi::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                           "the received is %d",
                                           quant_axis));
        });
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int &bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                            true,
                            phi::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
The scale of FakeChannelWiseQuantize operator is a vector.
In detail, each channel of the input X has a scale value.

$$scale_c = max(abs(X_c))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out_c = round(\frac{X_c * range} {scale_c})$$
In above three formulas, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$
)DOC");
  }
};

class FakeChannelWiseQuantizeDequantizeAbsMaxOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"),
                   "Input",
                   "X",
                   "FakeChannelWiseQuantizeDequantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FakeChannelWiseQuantizeDequantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"),
                   "Output",
                   "OutScale",
                   "FakeChannelWiseQuantizeDequantizeAbsMax");
    int quant_axis = ctx->Attrs().Get<int>("quant_axis");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {ctx->GetInputDim("X")[quant_axis]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class FakeChannelWiseQuantizeDequantizeAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized and dequantized low level tensor, "
              "saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current channel wise scale");
    AddAttr<int>("quant_axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0)
        .AddCustomChecker([](const int &quant_axis) {
          PADDLE_ENFORCE_EQ(
              quant_axis == 0 || quant_axis == 1,
              true,
              phi::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                           "the received is %d",
                                           quant_axis));
        });
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int &bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                            true,
                            phi::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddComment(R"DOC(
The scale of FakeChannelWiseQuantize operator is a vector.
In detail, each channel of the input X has a scale value.

$$scale_c = max(abs(X_c))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out_c = round(\frac{X_c * range} {scale_c}) * \frac{scale_c} {range}$$
In above three formulas, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$
)DOC");
  }
};

class MovingAverageAbsMaxScaleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "MovingAverageAbsMaxScale");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"),
                   "Output",
                   "OutScale",
                   "MovingAverageAbsMaxScale");

    if (ctx->HasOutput("OutState")) {
      ctx->SetOutputDim("OutState", {1});
    }
    if (ctx->HasOutput("OutAccum")) {
      ctx->SetOutputDim("OutAccum", {1});
    }
    if (ctx->HasOutput("Out")) {
      ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
      ctx->SetOutputDim("OutScale", {1});
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class MovingAverageAbsMaxScaleOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("InAccum", "Last accum.").AsDispensable();
    AddInput("InState", "Last state.").AsDispensable();
    AddOutput("Out",
              "(Tensor) Output tensor is just equivalent to the input tensor.")
        .AsDispensable();
    AddOutput("OutScale", " Current scale");
    AddOutput("OutState", "(Tensor) state buffer.").AsDispensable();
    AddOutput("OutAccum", "(Tensor) accum buffer.").AsDispensable();
    AddAttr<float>("moving_rate", "(float, default 0.9) moving rate.")
        .SetDefault(0.9);
    AddAttr<bool>("is_test",
                  "(bool, default false) Set true for inference only and false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
MovingAverageAbsMaxScale operator is only used for calculating the quantization scale.
And it will not quantize the input tensor.

$$scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)$$
$$Out = X$$

)DOC");
  }
};

class StraightThroughEstimatorGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    auto x_grad_name = framework::GradVarName("X");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name),
                   "Input",
                   out_grad_name,
                   "StraightThroughEstimatorGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(x_grad_name),
                   "Output",
                   x_grad_name,
                   "StraightThroughEstimatorGradOp");

    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim(out_grad_name));
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class StraightThroughEstimatorMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("straight_through_estimator_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = phi::CPUContext;

REGISTER_OPERATOR(
    fake_quantize_dequantize_abs_max,
    ops::FakeQuantOrWithDequantAbsMaxOp,
    ops::FakeQuantOrWithDequantAbsMaxOpMaker,
    ops::StraightThroughEstimatorMaker<paddle::framework::OpDesc>,
    ops::StraightThroughEstimatorMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(fake_quantize_dequantize_abs_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeDequantizeAbsMaxKernel,
                          float) {}

REGISTER_OPERATOR(
    fake_channel_wise_quantize_abs_max,
    ops::FakeChannelWiseQuantizeAbsMaxOp,
    ops::FakeChannelWiseQuantizeAbsMaxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(fake_channel_wise_quantize_abs_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::FakeChannelWiseQuantizeAbsMaxKernel,
                          float) {}

REGISTER_OPERATOR(
    moving_average_abs_max_scale,
    ops::MovingAverageAbsMaxScaleOp,
    ops::MovingAverageAbsMaxScaleOpMaker,
    ops::StraightThroughEstimatorMaker<paddle::framework::OpDesc>,
    ops::StraightThroughEstimatorMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(moving_average_abs_max_scale,
                          CPU,
                          ALL_LAYOUT,
                          ops::MovingAverageAbsMaxScaleKernel,
                          float) {}

REGISTER_OPERATOR(straight_through_estimator_grad,
                  ops::StraightThroughEstimatorGradOp);
PD_REGISTER_STRUCT_KERNEL(straight_through_estimator_grad,
                          CPU,
                          ALL_LAYOUT,
                          ops::StraightThroughEstimatorGradKernel,
                          float) {}

REGISTER_OPERATOR(
    fake_channel_wise_quantize_dequantize_abs_max,
    ops::FakeChannelWiseQuantizeDequantizeAbsMaxOp,
    ops::FakeChannelWiseQuantizeDequantizeAbsMaxOpMaker,
    ops::StraightThroughEstimatorMaker<paddle::framework::OpDesc>,
    ops::StraightThroughEstimatorMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(fake_channel_wise_quantize_dequantize_abs_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::FakeChannelWiseQuantizeDequantizeAbsMaxKernel,
                          float) {}

REGISTER_OP_VERSION(fake_channel_wise_quantize_abs_max)
    .AddCheckpoint(
        R"ROC(add new attributes [quant_axis] for applying per-channel "
        "quantization to conv2d_transpose and mul ops.)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "quant_axis", "The axis for quantization.", 0));
REGISTER_OP_VERSION(moving_average_abs_max_scale)
    .AddCheckpoint(
        R"ROC(Incompatible upgrade of output [Out])ROC",
        paddle::framework::compatible::OpVersionDesc().DeleteOutput(
            "Out",
            "Delete output in order to make the inference model not "
            "save moving_average_abs_max_scale operator. This will "
            "make the quantitative model be correctly applied in inference."))
    .AddCheckpoint(R"ROC(Incompatible upgrade of output [Out])ROC",
                   paddle::framework::compatible::OpVersionDesc().NewOutput(
                       "Out",
                       "In order to support dygraph qat, add output again."));
