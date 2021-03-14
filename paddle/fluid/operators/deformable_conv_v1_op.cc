// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/deformable_conv_v1_op.h"
#include <memory>
#include "paddle/fluid/operators/conv_op.h"

namespace paddle {
namespace operators {
class DeformableConvV1OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) The input of deformable conv op. "
             "The shape of input is "
             "[N, channel_in, H, W]");
    AddInput("Offset",
             "(Tensor) The input offset. "
             "The shape of the offset is "
             "[N, deformable_groups * kernel_w * kernel_h * 2, H, W");
    AddInput("Filter",
             "(Tensor) The Input Filter "
             "The shape of the wight is "
             "[num_filters, channel_in, kernel_h, kernel_w.");
    AddOutput("Output",
              "(Tensor) The output. "
              "The shape of the output tensor is "
              "[N, num_filters, out_height, out_width]].");
    AddAttr<std::vector<int>>("strides",
                              "(vector<int> default:{1, 1}), the "
                              "strides(h_stride, w_stride) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int> default:{0,0}), the "
                              "paddings(h_pad, w_pad) of "
                              "convolution operator. ")
        .SetDefault({0, 0});
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int> default:{1, 1}), the "
                              "dilations(h_dilation, w_dilation) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<int>(
        "groups",
        "(int default:1), the groups number of the convolution operator. "
        "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
        "when group=2, the first half of the filters is only connected to the "
        "first half of the input channels, while the second half of the "
        "filters "
        "is only connected to the second half of the input channels.")
        .SetDefault(1);
    AddAttr<int>("deformable_groups",
                 "(int default:1), the number of the deformable groups.")
        .SetDefault(1);
    AddAttr<int>("im2col_step",
                 "im2col maximum number of image per computation")
        .SetDefault(64);
    AddComment(R"DOC(
**Deformable Convolution v1 Operator**

Deformable Convolution is a new method based Convolution which feature has offset 
in spatial location.

1. Get offset of each pixel in feature map with convolution layers which number 
   of channels should be double of weight size.

2. Add offset to pixel to get new location and the new value which are computed 
   directly through bilinear interpolation with four nearest pixel.

3. Get the product of pixel and weight as result

Compute 2-D deformable convolution on 4-D input.

Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:

$$
y(p) = \\sum_{k=1}^{K}{w_k * x(p + p_k + \\Delta p_k)}
$$

Where $$\\Delta p_k$$ is the learnable offset for the k-th location, respectively.

Refer to 'https://arxiv.org/abs/1703.06211 '<https://arxiv.org/abs/1703.06211>

Example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, H_f, W_f)$
       Offset shape: $(N, 2 * deformable_groups, * H_f * W_f, H_{out}, W_{out})$
  Output:
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
                     where $H_{out}, W_{out}$ must be equal to $H_{in}, W_{in}$ respectively.
  Where
$$
       H_{out}= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]}+ 1 \\
       W_{out}= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]}+ 1
$$
)DOC");
  }
};

class DeformableConvV1Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input",
                   "deformable_conv_v1");
    OP_INOUT_CHECK(ctx->HasInput("Offset"), "Input", "Offset",
                   "deformable_conv_v1");
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter",
                   "deformable_conv_v1");
    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output",
                   "deformable_conv_v1");

    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    auto offset_dims = ctx->GetInputDim("Offset");

    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> dilations =
        ctx->Attrs().Get<std::vector<int>>("dilations");
    int groups = ctx->Attrs().Get<int>("groups");
    int deformable_groups = ctx->Attrs().Get<int>("deformable_groups");
    int im2col_step = ctx->Attrs().Get<int>("im2col_step");

    PADDLE_ENFORCE_EQ(
        in_dims.size(), 4,
        platform::errors::InvalidArgument(
            "Conv input should be 4-D tensor, get %u", in_dims.size()));
    PADDLE_ENFORCE_EQ(in_dims.size(), filter_dims.size(),
                      platform::errors::InvalidArgument(
                          "Conv input dimension and filter dimension should be "
                          "the same. the difference is [%d] vs [%d]",
                          in_dims.size(), filter_dims.size()));
    PADDLE_ENFORCE_EQ(
        in_dims.size() - strides.size(), 2U,
        platform::errors::InvalidArgument(
            "Conv input dimension and strides "
            "dimension should be consistent., But received [%d]: [%d]",
            in_dims.size(), strides.size()));
    PADDLE_ENFORCE_EQ(paddings.size(), strides.size(),
                      platform::errors::InvalidArgument(
                          "Conv paddings dimension and Conv strides dimension "
                          "should be the same. The difference is [%d] vs [%d]",
                          paddings.size(), strides.size()));

    PADDLE_ENFORCE_EQ(
        in_dims[1], filter_dims[1] * groups,
        platform::errors::InvalidArgument(
            "The number of input channels should be equal to filter "
            "channels * groups. The difference is [%d]: [%d]",
            in_dims[1], filter_dims[1] * groups));
    PADDLE_ENFORCE_EQ(
        filter_dims[0] % groups, 0,
        platform::errors::InvalidArgument(
            "The number of output channels should be divided by groups. But"
            "received output channels: [%d], groups: [%d]",
            filter_dims[0], groups));
    PADDLE_ENFORCE_EQ(
        filter_dims[0] % deformable_groups, 0,
        platform::errors::InvalidArgument(
            "The number of output channels should be "
            "divided by deformable groups. But received [%d]: [%d]",
            filter_dims[0], deformable_groups));

    if (in_dims[0] > im2col_step) {
      PADDLE_ENFORCE_EQ(in_dims[0] % im2col_step, 0U,
                        platform::errors::InvalidArgument(
                            "Input batchsize must be smaller than or divide "
                            "im2col_step, But received [%d]: [%d]",
                            in_dims[0], im2col_step));
    }

    for (size_t i = 0; i < strides.size(); ++i) {
      PADDLE_ENFORCE_GT(strides[i], 0U, platform::errors::InvalidArgument(
                                            "stride %d size incorrect", i));
    }
    for (size_t i = 0; i < dilations.size(); ++i) {
      PADDLE_ENFORCE_GT(dilations[i], 0U, platform::errors::InvalidArgument(
                                              "dilation %d size incorrect", i));
    }

    std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
    for (size_t i = 0; i < strides.size(); ++i) {
      if ((!ctx->IsRuntime()) &&
          (in_dims[i + 2] <= 0 || filter_dims[i + 2] <= 0)) {
        output_shape.push_back(-1);
      } else {
        output_shape.push_back(ConvOutputSize(in_dims[i + 2],
                                              filter_dims[i + 2], dilations[i],
                                              paddings[i], strides[i]));
      }
    }
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(output_shape[1] % deformable_groups, 0U,
                        platform::errors::InvalidArgument(
                            "output num_filter must divide deformable group "
                            "size. But received [%d]: [%d]",
                            output_shape[1], deformable_groups));
      PADDLE_ENFORCE_EQ(output_shape[2], offset_dims[2],
                        platform::errors::InvalidArgument(
                            "output height must equal to offset map height. "
                            "The difference is [%d]: [%d]",
                            output_shape[2], offset_dims[2]));
      PADDLE_ENFORCE_EQ(output_shape[3], offset_dims[3],
                        platform::errors::InvalidArgument(
                            "output width must equal to offset map width. The "
                            "difference is [%d]: [%d]",
                            output_shape[3], offset_dims[3]));
      PADDLE_ENFORCE_EQ(offset_dims[1] % (filter_dims[2] * filter_dims[3]), 0U,
                        platform::errors::InvalidArgument(
                            "offset filter must divide deformable group size. "
                            "But received [%d]: [%d]",
                            offset_dims[1], filter_dims[2] * filter_dims[3]));
      PADDLE_ENFORCE_EQ(
          offset_dims[1] / (2 * filter_dims[2] * filter_dims[3]),
          deformable_groups,
          platform::errors::InvalidArgument(
              "offset filter must divide deformable group size. But received "
              "[%d]: [%d]",
              offset_dims[1] / (2 * filter_dims[2] * filter_dims[3]),
              deformable_groups));
    }
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

template <typename T>
class DeformableConvV1GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("deformable_conv_v1_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetInput("Offset", this->Input("Offset"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));
    op->SetOutput(framework::GradVarName("Offset"), this->InputGrad("Offset"));

    op->SetAttrMap(this->Attrs());
  }
};

class DeformableConvV1GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    auto offset_dims = ctx->GetInputDim("Offset");

    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Output")), "Input",
                   "Output@Grad", "deformable_conv_v1_grad");
    if (ctx->HasOutput(framework::GradVarName("Input"))) {
      ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Offset"))) {
      ctx->SetOutputDim(framework::GradVarName("Offset"), offset_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(deformable_conv_v1, ops::DeformableConvV1Op,
                  ops::DeformableConvV1OpMaker,
                  ops::DeformableConvV1GradOpMaker<paddle::framework::OpDesc>,
                  ops::DeformableConvV1GradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(deformable_conv_v1_grad, ops::DeformableConvV1GradOp);

REGISTER_OP_CPU_KERNEL(deformable_conv_v1,
                       ops::DeformableConvV1CPUKernel<float>);
REGISTER_OP_CPU_KERNEL(deformable_conv_v1_grad,
                       ops::DeformableConvV1GradCPUKernel<float>);
