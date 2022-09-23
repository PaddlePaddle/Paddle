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

#include "paddle/fluid/operators/deformable_psroi_pooling_op.h"

#include <iostream>
#include <memory>
#include <vector>

#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {
class DeformablePSROIPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor), "
             "the input of Deformable PSROIPooling. "
             "The shape of input tensor is [N,C,H,W]. Where N is batch size, "
             "C is number of input channels, "
             "H is height of the feature, and "
             "W is the width of the feature.");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "ROIs should be a 2-D LoDTensor of shape (num_rois, 4) "
             "given as [[x1, y1, x2, y2], ...]. "
             "(x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates.");
    AddInput("Trans",
             "(Tensor),"
             "offset of features on ROIs while pooling. "
             "The format is NCHW, where N is number of ROIs, "
             "C is number of channels, which indicate the offset distance "
             "in the x and y directions, "
             "H is pooled height, and "
             "W is pooled width.");
    AddAttr<bool>("no_trans",
                  "(bool), "
                  "whether add offset to get new value or not while roi "
                  "pooling, which value is True or False");
    AddAttr<float>("spatial_scale",
                   "(float), "
                   "ratio of input feature map height (or width) to "
                   "raw image height (or width). Equals the reciprocal "
                   "of total stride in convolutional layers.");
    AddAttr<int>("output_dim",
                 "(int), "
                 "the number of output channels, which should be less than "
                 "input channels. Deformable roi_pooling requires "
                 "output_channels = input_channels, while deformable "
                 "psroi_pooling requires output_channels = input_channels "
                 "* pooled_height * pooled_width");
    AddAttr<std::vector<int>>(
        "group_size",
        "(vector<int>), "
        "the number of groups which input channels are divided."
        "(eg.number of input channels is k1*k2*(C+1), which k1 and k2 "
        "are group width and height and C+1 is number of output "
        "channels. eg.(4, 6), which 4 is height of group and 6 is "
        "width of group");
    AddAttr<int>("pooled_height",
                 "(int), "
                 "the pooled output height.");
    AddAttr<int>("pooled_width",
                 "(int), "
                 "the pooled output width.");
    AddAttr<std::vector<int>>(
        "part_size",
        "(vector<int>), "
        "the height and width of offset, eg.(4, 6), which height is 4 "
        " and width is 6");
    AddAttr<int>("sample_per_part",
                 "(int), "
                 "the number of samples in each bin");
    AddAttr<float>("trans_std",
                   "(float), "
                   "Coefficient of offset");
    AddOutput("TopCount",
              "(Tensor), "
              "record the number of pixel in average pooling to in each bin. "
              "The format is NCHW, where N is the number of ROIs, "
              "C is the number of output channels, "
              "H is the height of output, and "
              "W is the width of output.");
    AddOutput("Output",
              "(Tensor), "
              "the output of Deformable PSROIPooling. "
              "The format is NCHW, where N is the number of ROIs, "
              "C is the number of output channels, "
              "H is the height of output, and "
              "W is thewidth of output. ");
    AddComment(R"DOC(
**DeformablePSROIPooling Operator**
DeformablePSROIPooling is a new method based Region of interest pooling
(also known as RoI pooling).
The operator has four steps:

1. Dividing each region proposal into equal-sized sections with
   the pooled_width and pooled_height.

2. Add offset to pixel in ROI to get new location and the new value which are
   computed directly through bilinear interpolation with four nearest pixel.

3. Sample several points to get average values in each bin.

4. Copying these average values to the output buffer.

DeformablePSROIPooling is part of Deformable Convolutional Networks,
please refer to https://arxiv.org/abs/1703.06211 for more details.
    )DOC");
  }
};

class DeformablePSROIPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Input"), "Input", "Input", "deformable_psroi_pooling");
    OP_INOUT_CHECK(
        ctx->HasInput("ROIs"), "Input", "ROIs", "deformable_psroi_pooling");
    OP_INOUT_CHECK(
        ctx->HasInput("Trans"), "Input", "Trans", "deformable_psroi_pooling");
    OP_INOUT_CHECK(ctx->HasOutput("Output"),
                   "Output",
                   "Output",
                   "deformable_psroi_pooling");
    OP_INOUT_CHECK(ctx->HasOutput("TopCount"),
                   "Output",
                   "TopCount",
                   "deformable_psroi_pooling");
    auto input_dims = ctx->GetInputDim("Input");
    auto rois_dims = ctx->GetInputDim("ROIs");
    auto trans_dims = ctx->GetInputDim("Trans");
    PADDLE_ENFORCE_EQ(
        rois_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Input(ROIs) should be a 2-D LoDTensor of shape (num_rois, 4) "
            "given as [[ x1, y1, x2, y2], ...]. The rank of Input(ROIs) should "
            "be 2, but received ROIs rank is:%d, ROIs shape is:[%s].",
            rois_dims.size(),
            rois_dims));
    PADDLE_ENFORCE_EQ(
        trans_dims.size(),
        4,
        platform::errors::InvalidArgument("The rank of Input(Trans) should be "
                                          "4 and the shape of Trans should be "
                                          "(N, 2, H, W), but received Trans "
                                          "rank is:%d and Trans shape is:[%s].",
                                          trans_dims.size(),
                                          trans_dims));
    auto pooled_height = ctx->Attrs().Get<int>("pooled_height");
    auto pooled_width = ctx->Attrs().Get<int>("pooled_width");
    auto spatial_scale = ctx->Attrs().Get<float>("spatial_scale");
    auto output_channels = ctx->Attrs().Get<int>("output_dim");
    auto group_size = ctx->Attrs().Get<std::vector<int>>("group_size");
    auto group_height = group_size[0];
    auto group_width = group_size[1];
    auto part_size = ctx->Attrs().Get<std::vector<int>>("part_size");
    auto part_height = part_size[0];
    auto part_width = part_size[1];
    auto sample_per_part = ctx->Attrs().Get<int>("sample_per_part");
    auto trans_std = ctx->Attrs().Get<float>("trans_std");
    PADDLE_ENFORCE_GE(trans_std,
                      0.,
                      platform::errors::InvalidArgument(
                          "Input(trans_std) should not be lower "
                          "than 0.0, but received trans_std "
                          "is:%f",
                          trans_std));
    PADDLE_ENFORCE_GE(
        input_dims[1],
        output_channels,
        platform::errors::InvalidArgument(
            "The channel of Input(Input) should not be lower than "
            "Input(output_dim), "
            "but received Input channel is:%d and output_dim is:%d.",
            input_dims[1],
            output_channels));
    PADDLE_ENFORCE_GT(
        pooled_height,
        0,
        platform::errors::InvalidArgument(
            "Input(pooled_height) should be greater than 0, but received "
            "pooled_height is:%d.",
            pooled_height));
    PADDLE_ENFORCE_GT(
        pooled_width,
        0,
        platform::errors::InvalidArgument(
            "Input(pooled_width) should be greater than 0, but received "
            "pooled_width is:%d.",
            pooled_width));
    PADDLE_ENFORCE_GT(
        spatial_scale,
        0.,
        platform::errors::InvalidArgument(
            "Input(spatial_scale) should be greater than 0., but received "
            "spatial_scale is:%f.",
            spatial_scale));
    PADDLE_ENFORCE_EQ(
        group_size.size(),
        2,
        platform::errors::InvalidArgument(
            "The length of Input(group_size) should be 2, but received "
            "group_size length is:%d.",
            group_size.size()));
    PADDLE_ENFORCE_GT(
        group_height,
        0,
        platform::errors::InvalidArgument(
            "group_height in Input(group_size) should be greater than 0, "
            "but received group_height is:%d.",
            group_height));
    PADDLE_ENFORCE_GT(
        group_width,
        0,
        platform::errors::InvalidArgument(
            "group_width in Input(group_size) should be greater than 0 "
            "but received group_width is:%d.",
            group_width));
    PADDLE_ENFORCE_EQ(
        part_size.size(),
        2,
        platform::errors::InvalidArgument(
            "The length of Input(part_size) should be 2, but received "
            "part_size length is:%d.",
            part_size.size()));
    PADDLE_ENFORCE_GT(
        part_height,
        0,
        platform::errors::InvalidArgument(
            "part_height in Input(part_size) should be greater than 0 "
            "but received part_height is:%d.",
            part_height));
    PADDLE_ENFORCE_GT(
        part_width,
        0,
        platform::errors::InvalidArgument(
            "part_width in Input(part_size) should be greater than 0 "
            "but received part_width is:%d.",
            part_width));
    PADDLE_ENFORCE_LE(
        part_height,
        trans_dims[2],
        platform::errors::InvalidArgument(
            "part_height in Input(part_size) should not be greater than "
            "the height of Input(Trans), but received part_height is:%d, "
            "the height of Input(Trans) is:%d.",
            part_height,
            trans_dims[2]));
    PADDLE_ENFORCE_LE(
        part_width,
        trans_dims[3],
        platform::errors::InvalidArgument(
            "part_width in Input(part_size) should not be greater than "
            "the width of Input(Trans), but received part_width is:%d, "
            "the width of Input(Trans) is:%d.",
            part_width,
            trans_dims[3]));
    PADDLE_ENFORCE_GT(
        sample_per_part,
        0,
        platform::errors::InvalidArgument(
            "Input(sample_per_part) should be greater than 0, but received "
            "sample_per_part is:%d.",
            sample_per_part));
    auto out_dims = input_dims;
    out_dims[0] = rois_dims[0];
    out_dims[1] = output_channels;
    out_dims[2] = pooled_height;
    out_dims[3] = pooled_width;
    ctx->SetOutputDim("Output", out_dims);
    ctx->SetOutputDim("TopCount", out_dims);
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
class DeformablePSROIPoolGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("deformable_psroi_pooling_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Trans", this->Input("Trans"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("TopCount", this->Output("TopCount"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Trans"), this->InputGrad("Trans"));

    op->SetAttrMap(this->Attrs());
  }
};

class DeformablePSROIPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Output")),
                   "Input",
                   "Output@GRAD",
                   "deformable_psroi_pooling");
    if (ctx->HasOutput(framework::GradVarName("Input"))) {
      ctx->SetOutputDim(framework::GradVarName("Input"),
                        ctx->GetInputDim("Input"));
    }
    if (ctx->HasOutput(framework::GradVarName("Trans"))) {
      ctx->SetOutputDim(framework::GradVarName("Trans"),
                        ctx->GetInputDim("Trans"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Trans"),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = phi::CPUContext;
REGISTER_OPERATOR(
    deformable_psroi_pooling,
    ops::DeformablePSROIPoolOp,
    ops::DeformablePSROIPoolOpMaker,
    ops::DeformablePSROIPoolGradOpMaker<paddle::framework::OpDesc>,
    ops::DeformablePSROIPoolGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(deformable_psroi_pooling_grad,
                  ops::DeformablePSROIPoolGradOp);
REGISTER_OP_CPU_KERNEL(deformable_psroi_pooling,
                       ops::DeformablePSROIPoolCPUKernel<CPU, float>,
                       ops::DeformablePSROIPoolCPUKernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(deformable_psroi_pooling_grad,
                       ops::DeformablePSROIPoolGradCPUKernel<CPU, float>,
                       ops::DeformablePSROIPoolGradCPUKernel<CPU, double>);
