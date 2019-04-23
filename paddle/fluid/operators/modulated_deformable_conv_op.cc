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

#include "paddle/fluid/operators/conv_op.h"

namespace paddle {
namespace operators {
class ModulatedDeformableConvOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) The input of modulated deformable conv op. "
             "The format of input is NCHW");
    AddInput("Offset",
             "(Tensor) The input offset. "
             "The shape of the offset is "
             "[N, groups * kernel_w * kernel_h * 2, H, W");
    AddInput("Mask",
             "(Tensor) The input mask. "
             "The shape of the mask is "
             "[N, groups * kernel_w * kernel_h, H, W].");
    AddInput("Filter",
             "(Tensor) The Input Filter "
             "The shape of the wight is "
             "[num_filters, channel_input, kernel_h, kernel_w.");
    AddInput("Bias",
             "(Tensor) The Input Bias "
             "The shape of the bias is "
             "[num_filters, ].")
        .AsDispensable();
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
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int> default:{1, 1}), the "
                              "dilations(h_dilation, w_dilation) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<int>("im2col_step",
                 "im2col maximum number of image per computation")
        .SetDefault(64);
    AddComment(R"DOC(
**Modulated Deformable Convolution Operator**

https://arxiv.org/abs/1811.11168
)DOC");
  }
};

class ModulatedDeformableConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of ModulatedDeformableConvOp "
                   "should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Offset"),
                   "Input(Offset) of ModulatedDeformableConvOp "
                   "should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Mask"),
                   "Input(Mask) of ModulatedDeformableConvOp "
                   "should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Filter"),
                   "Input(Filter) of ModulatedDeformableConvOp "
                   "should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Output"),
                   "Output(Output) of ModulatedDeformableConvOp "
                   "should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    auto offset_dims = ctx->GetInputDim("Offset");
    auto mask_dims = ctx->GetInputDim("Mask");

    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> dilations =
        ctx->Attrs().Get<std::vector<int>>("dilations");
    int groups = ctx->Attrs().Get<int>("groups");
    int deformable_groups = ctx->Attrs().Get<int>("deformable_groups");
    int im2col_step = ctx->Attrs().Get<int>("im2col_step");

    PADDLE_ENFORCE(in_dims.size() == 4,
                   "Conv intput should be 4-D tensor, get %u", in_dims.size());
    PADDLE_ENFORCE_EQ(
        in_dims.size(), filter_dims.size(),
        "Conv input dimension and filter dimension should be the same.");
    PADDLE_ENFORCE_EQ(
        in_dims.size() - strides.size(), 2U,
        "Conv input dimension and strides dimension should be consistent.");
    PADDLE_ENFORCE_EQ(paddings.size(), strides.size(),
                      "Conv paddings dimension and Conv strides dimension "
                      "should be the same.");

    PADDLE_ENFORCE_EQ(in_dims[1], filter_dims[1] * groups,
                      "The number of input channels should be equal to filter "
                      "channels * groups.");
    PADDLE_ENFORCE_EQ(
        filter_dims[0] % groups, 0,
        "The number of output channels should be divided by groups.");
    PADDLE_ENFORCE_EQ(filter_dims[0] % deformable_groups, 0,
                      "The number of output channels should be "
                      "divided by deformable groups.");

    if (in_dims[0] > im2col_step) {
      PADDLE_ENFORCE_EQ(
          in_dims[0] % im2col_step, 0U,
          "Input batchsize must be smaller than or divide im2col_step");
    }

    for (size_t i = 0; i < strides.size(); ++i) {
      PADDLE_ENFORCE_GT(strides[i], 0U, "incorrect stride size");
    }
    for (size_t i = 0; i < paddings.size(); ++i) {
      PADDLE_ENFORCE_GT(paddings[i], 0U, "incorrect padding size");
    }
    for (size_t i = 0; i < dilations.size(); ++i) {
      PADDLE_ENFORCE_GT(dilations[i], 0U, "incorrect dilation size");
    }

    std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
    for (size_t i = 0; i < strides.size(); ++i) {
      output_shape.push_back(ConvOutputSize(in_dims[i + 2], filter_dims[i + 2],
                                            dilations[i], paddings[i],
                                            strides[i]));
    }
    PADDLE_ENFORCE_EQ(output_shape[1] % deformable_groups, 0U,
                      "output num_filter must divide deformable group size.");
    PADDLE_ENFORCE_EQ(output_shape[2], offset_dims[2],
                      "output height must equal to offset map height.");
    PADDLE_ENFORCE_EQ(output_shape[3], offset_dims[3],
                      "output width must equal to offset map width.");
    PADDLE_ENFORCE_EQ(offset_dims[1] % (filter_dims[2] * filter_dims[3]), 0U,
                      "offset filter must divide deformable group size.");
    PADDLE_ENFORCE_EQ(offset_dims[1] / (2 * filter_dims[2] * filter_dims[3]),
                      0U, "offset filter must divide deformable group size.");
    PADDLE_ENFORCE_EQ(output_shape[2], mask_dims[2],
                      "output height must equal to mask map height.");
    PADDLE_ENFORCE_EQ(output_shape[3], mask_dims[3],
                      "output width must equal to mask map width.");
    PADDLE_ENFORCE_EQ(mask_dims[1] % (filter_dims[2] * filter_dims[3]), 0U,
                      "mask filter must divide deformable group size.");
    PADDLE_ENFORCE_EQ(mask_dims[1] / (filter_dims[2] * filter_dims[3]), 0U,
                      "mask filter must divide deformable group size.");

    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
    // TODO(yifan): Add share LOD
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.device_context());
  }
};

class ModulatedDeformableConvGradOpDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());

    op->SetType("modulated_deformable_conv_grad");
    op->SetInput("Input", Input("Input"));
    op->SetInput("Filter", Input("Filter"));
    op->SetInput("Bias", Input("Bias"));
    op->SetInput("Offset", Input("Offset"));
    op->SetInput("Mask", Input("Mask"));
    op->SetInput(framework::GradVarName("Output"), OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Filter"), InputGrad("Filter"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));
    op->SetOutput(framework::GradVarName("Offset"), InputGrad("Offset"));
    op->SetOutput(framework::GradVarName("Mask"), InputGrad("Mask"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

class ModulatedDeformableConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    auto offset_dims = ctx->GetInputDim("Offset");
    auto mask_dims = ctx->GetInputDim("Mask");

    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Output")),
                   "the gradient of output(Out) must not be null");
    if (ctx->HasOutput(framework::GradVarName("Input"))) {
      ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Offset"))) {
      ctx->SetOutputDim(framework::GradVarName("Offset"), offset_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Mask"))) {
      ctx->SetOutputDim(framework::GradVarName("Mask"), mask_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(modulated_deformable_conv, ops::ModulatedDeformableConvOp,
                  ops::ModulatedDeformableConvOpMaker,
                  ops::ModulatedDeformableConvGradOpDescMaker);

REGISTER_OPERATOR(modulated_deformable_conv_grad,
                  ops::ModulatedDeformableConvGradOp);

// REGISTER_OP_CPU_KERNEL(
//     modulated_deformable_conv,
//     ops::ModulatedDeformableConvKernel<paddle::platform::CPUDeviceContext,
//                                        float>,
//     ops::ModulatedDeformableConvKernel<paddle::platform::CPUDeviceContext,
//                                        double>);

// REGISTER_OP_CPU_KERNEL(
//     modulated_deformable_conv_grad,
//     ops::ModulatedDeformableConvGradKernel<paddle::platform::CPUDeviceContext,
//                                            float>,
//     ops::ModulatedDeformableConvGradKernel<paddle::platform::CPUDeviceContext,
//                                            double>);
