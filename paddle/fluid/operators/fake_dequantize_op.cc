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

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

template <typename T>
struct DequantizeFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  phi::DenseTensor* out) {
    auto in_e = framework::EigenVector<T>::Flatten(*in);
    const T* scale_factor = scale->data<T>();
    auto out_e = framework::EigenVector<T>::Flatten(*out);

    auto& dev = *dev_ctx.eigen_device();
    out_e.device(dev) = in_e * scale_factor[0] / max_range;
  }
};

template <typename T>
struct ChannelDequantizeFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor** scales,
                  const int scale_num,
                  T max_range,
                  const int quant_axis,
                  const int x_num_col_dims,
                  phi::DenseTensor* out) {
    if (scale_num == 1) {
      // Dequant op is before quantized op
      // Dequantize the weight of quantized op
      auto in_dims = in->dims();
      const int64_t channel = in_dims[quant_axis];
      const T* scale_factor = scales[0]->data<T>();
      if (quant_axis == 0) {
        for (int64_t i = 0; i < channel; i++) {
          T s = scale_factor[i];
          phi::DenseTensor one_channel_in = in->Slice(i, i + 1);
          phi::DenseTensor one_channel_out = out->Slice(i, i + 1);
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
    } else if (scale_num == 2) {
      // Dequant op is after quantized op
      // Dequantize the output tensor of quantized op
      if (x_num_col_dims > 1) {
        auto in_dims = in->dims();
        const int64_t channel = in_dims[x_num_col_dims];
        const T* scale_one = scales[0]->data<T>();
        const T* scale_two = scales[1]->data<T>();
        int64_t out_iter = 1;
        for (int i = 0; i < x_num_col_dims; i++) {
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
            T s = scale_one[j];
            for (int64_t k = 0; k < step_j; k++) {
              *cur_out = (*cur_in) * s * scale_two[0] / max_range;
              ++cur_in;
              ++cur_out;
            }
          }
        }
      } else {
        int batch_size = in->dims()[0];
        int channel = in->dims()[1];
        const T* scale_one = scales[0]->data<T>();
        const T* scale_two = scales[1]->data<T>();
        for (int i = 0; i < batch_size; i++) {
          phi::DenseTensor one_batch_in = in->Slice(i, i + 1).Resize(
              phi::slice_ddim(in->dims(), 1, in->dims().size()));
          phi::DenseTensor one_batch_out = out->Slice(i, i + 1).Resize(
              phi::slice_ddim(out->dims(), 1, out->dims().size()));
          for (int j = 0; j < channel; j++) {
            T s = scale_one[j];
            phi::DenseTensor one_channel_in = one_batch_in.Slice(j, j + 1);
            phi::DenseTensor one_channel_out = one_batch_out.Slice(j, j + 1);
            auto in_e = framework::EigenVector<T>::Flatten(one_channel_in);
            auto out_e = framework::EigenVector<T>::Flatten(one_channel_out);
            auto& dev = *dev_ctx.eigen_device();
            out_e.device(dev) = in_e * s * scale_two[0] / max_range;
          }
        }
      }
    }
  }
};

template struct DequantizeFunctor<phi::CPUContext, float>;
template struct DequantizeFunctor<phi::CPUContext, double>;
template struct ChannelDequantizeFunctor<phi::CPUContext, float>;
template struct ChannelDequantizeFunctor<phi::CPUContext, double>;

class FakeDequantizeMaxAbsOp : public framework::OperatorWithKernel {
 public:
  FakeDequantizeMaxAbsOp(const std::string& type,
                         const framework::VariableNameMap& inputs,
                         const framework::VariableNameMap& outputs,
                         const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FakeDequantizeMaxAbs");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "FakeDequantizeMaxAbs");

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
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "FakeChannelWiseDequantizeMaxAbs");
    OP_INOUT_CHECK(ctx->HasInputs("Scales"),
                   "Input",
                   "Scales",
                   "FakeChannelWiseDequantizeMaxAbs");
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FakeChannelWiseDequantizeMaxAbs");

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
    AddAttr<int>("quant_axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0)
        .AddCustomChecker([](const int& quant_axis) {
          PADDLE_ENFORCE_EQ(quant_axis == 0 || quant_axis == 1,
                            true,
                            platform::errors::InvalidArgument(
                                "'quant_axis' should be 0 or 1, but "
                                "the received is %d",
                                quant_axis));
        });
    AddAttr<int>("x_num_col_dims",
                 "The x_num_col_dims of mul. Only used for mul or matmul.")
        .SetDefault(1)
        .AddCustomChecker([](const int& x_num_col_dims) {
          PADDLE_ENFORCE_EQ(x_num_col_dims == 0,
                            false,
                            platform::errors::InvalidArgument(
                                "'x_num_col_dims' should be larger than 0, but "
                                "the received is %d",
                                x_num_col_dims));
        });
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
using CPU = phi::CPUContext;

REGISTER_OPERATOR(
    fake_dequantize_max_abs,
    ops::FakeDequantizeMaxAbsOp,
    ops::FakeDequantizeMaxAbsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fake_dequantize_max_abs,
                       ops::FakeDequantizeMaxAbsKernel<CPU, float>,
                       ops::FakeDequantizeMaxAbsKernel<CPU, double>);

REGISTER_OPERATOR(
    fake_channel_wise_dequantize_max_abs,
    ops::FakeChannelWiseDequantizeMaxAbsOp,
    ops::FakeChannelWiseDequantizeMaxAbsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fake_channel_wise_dequantize_max_abs,
                       ops::FakeChannelWiseDequantizeMaxAbsKernel<CPU, float>,
                       ops::FakeChannelWiseDequantizeMaxAbsKernel<CPU, double>);

REGISTER_OP_VERSION(fake_channel_wise_dequantize_max_abs)
    .AddCheckpoint(
        R"ROC(add new attributes [quant_axis] for applying per-channel "
        "dequantization to conv2d_tranpose and mul ops.)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "quant_axis", "The axis for dequantization.", 0))
    .AddCheckpoint(
        R"ROC(add new attributes [x_num_col_dims] for applying per-channel "
        "dequantization to mul ops.)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "x_num_col_dims", "The x_num_col_dims for dequantization.", 1));
