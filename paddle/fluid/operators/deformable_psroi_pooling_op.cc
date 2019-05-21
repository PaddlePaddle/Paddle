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

#include <iostream>
#include "paddle/fluid/operators/deformable_psroi_pooling_op.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
class DeformablePSROIPoolOpMaker: public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor), "
             "the input of Deformable PSROIPooling. "
             "The format of input tensor is NCHW. Where N is batch size, "
             "C is number of input channels, "
             "H is height of the feature, and "
             "W is the width of the feature.");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "Should be a 2-D LoDTensor of shape (num_rois, 4) "
             "given as [[x1, y1, x2, y2], ...]. "
             "(x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates.");
    AddInput("Trans",
             "(Tensor),"
             "offset of features on ROIs while pooling. "
             "The format is NCHW, "
             "N is number of ROIs, "
             "C is the distance of offset in x and y, "
             "H is pooled height, "
             "W is pooled width.");
    AddAttr<int>("no_trans",
                 "(int), "
                 "Whether add offset to get new value or not while pooling, "
                 "value is 0 or 1").SetDefault(0);
    AddAttr<float>("spatial_scale",
                   "(float), "
                   "Ratio of input feature map height (or width) to " 
                   "raw image height (or width). Equals the reciprocal " 
                   "of total stride in convolutional layers.").SetDefault(1.0);
    AddAttr<int>("output_dim",
                 "(int), "
                 "number of output channels.").SetDefault(256);
    AddAttr<int>("group_size",
                 "(int), "
                 "number of groups which the channel is divided. ")
        .SetDefault(1);
    AddAttr<int>("pooled_size",
                 "(int), "
                 "output size which height is equal to width.").SetDefault(7);
    AddAttr<int>("part_size",
                 "(int), "
                 "height(or width) of offset which height is equal to width.")
        .SetDefault(7);
    AddAttr<int>("sample_per_part",
                 "(int), "
                 "number of samples in each bin").SetDefault(4);
    AddAttr<float>("trans_std",
                   "(float), "
                   "coefficient of offset").SetDefault(0.1);
    AddOutput("TopCount",
              "(Tensor), "
              "record the number of pixel in average pooling to in each bin, "
              "the format is NCHW, "
              "N is number of batch size, "
              "C is number of channel of output, "
              "H is height of output, "
              "W is width of output.");
    AddOutput("Output",
              "(Tensor), "
              "the output of Deformable PSROIPooling, "
              "the format is NCHW, where N is number of ROIs, "
              "C is number of output channels, "
              "H is height of output, "
              "W is width of output. ");
    AddComment(R"DOC(
**Deformable ps roi pooling Operator**
DeformablePSROIPooling is a new method based Region of interest pooling 
(also known as RoI pooling).
The operator has four steps:

1. Dividing each region proposal into equal-sized sections with
   the pooled_width and pooled_height.

2. Add offset to pixel in ROI to get new location and the new value are
   computed directly through bilinear interpolation with four nearest pixel.

3. Sample several points to get average values in each bin.

4. Copying these average values to the output buffer.

DeformablePSROIPooling is part of Deformable Convolutional Networksï
please refer to https://arxiv.org/abs/1703.06211 for more details)DOC");
  }
};

class DeformablePSROIPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of DeformablePSROIPoolOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("ROIs"),
                   "Input(ROIs) of DeformablePSROIPoolOp "                    
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Trans"),
                   "Input(Trans) of DeformablePSROIPoolOp "
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Output"),
                   "Output(Output) of DeformablePSROIPoolOp "
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("TopCount"),
                   "Output(TopCount) of DeformablePSROIPoolOp "
                   "should not be null.");
    auto input_dims = ctx->GetInputDim("Input"); 
    auto rois_dims = ctx->GetInputDim("ROIs");
    auto trans_dims = ctx->GetInputDim("Trans");
    PADDLE_ENFORCE(rois_dims.size() == 2,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 4)"
                   "given as [[ x1, y1, x2, y2], ...].");
    PADDLE_ENFORCE(trans_dims.size() == 4,
                   "The format of Input Trans is (N, 2, H, W).");
    auto pooled_size = ctx->Attrs().Get<int>("pooled_size");
    auto spatial_scale = ctx->Attrs().Get<float>("spatial_scale");
    auto output_dim = ctx->Attrs().Get<int>("output_dim");
    auto group_size = ctx->Attrs().Get<int>("group_size");
    auto part_size = ctx->Attrs().Get<int>("part_size");
    auto sample_per_part = ctx->Attrs().Get<int>("sample_per_part");
    auto trans_std = ctx->Attrs().Get<float>("trans_std");
    PADDLE_ENFORCE(trans_std >= 0.0,
                   "trans_std must be greater than 0.0");
    PADDLE_ENFORCE_GT(pooled_size, 0,
                      "The pooled size must greater than 0");
    PADDLE_ENFORCE_GT(spatial_scale, 0.0f,
                      "The spatial scale must greater than 0");
    PADDLE_ENFORCE_GT(group_size, 0,
                      "The group_size must greater than 0");
    PADDLE_ENFORCE_GT(part_size, 0,
                      "The part_size must greater than 0");
    PADDLE_ENFORCE_GT(sample_per_part, 0,
                      "The sample_per_part must greater than 0");
    auto out_dims = input_dims;
    out_dims[0] = rois_dims[0];
    out_dims[1] = output_dim;
    out_dims[2] = pooled_size;
    out_dims[3] = pooled_size;
    ctx->SetOutputDim("Output", out_dims);
    ctx->SetOutputDim("TopCount", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.device_context());
  }
};

class DeformablePSROIPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Output")),
                   "The gradient of Output should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Input")),
                   "The gradient of Input should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Trans")),
                   "The gradient of Trans should not be null.");
    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
    ctx->SetOutputDim(framework::GradVarName("Trans"),
                      ctx->GetInputDim("Trans"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Trans")->type(),
                                   ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(deformable_psroi_pooling, ops::DeformablePSROIPoolOp,
                  ops::DeformablePSROIPoolOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OPERATOR(deformable_psroi_pooling_grad,
                  ops::DeformablePSROIPoolGradOp);

REGISTER_OP_CPU_KERNEL(deformable_psroi_pooling,
    ops::DeformablePSROIPoolCPUKernel<CPU, float>,
    ops::DeformablePSROIPoolCPUKernel<CPU, double>);

REGISTER_OP_CPU_KERNEL(deformable_psroi_pooling_grad,
    ops::DeformablePSROIPoolGradCPUKernel<CPU, float>,
    ops::DeformablePSROIPoolGradCPUKernel<CPU, double>);

