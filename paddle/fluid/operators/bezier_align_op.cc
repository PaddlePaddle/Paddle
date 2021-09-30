/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/bezier_align_op.h"
#include <memory>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class BezierAlignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, "
             "the input of BezierAlignOp. "
             "The format of input tensor is NCHW. Where N is the batch size, "
             "C is the number of input channels, "
             "H is the height of the input feature map, and "
             "W is the width. The data type can be float32 or float64");
    AddInput("ROIs",
             "Tensor, "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D Tensor of shape (num_rois, 16) "
             "given as [(x1, y1, x2, y2), ...]. "
             "where (x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates. "
             "The roi batch index can be calculated from LoD.");
    AddInput("RoisNum",
             "(Tensor), "
             "The number of RoIs in each image.");
    AddOutput("Out",
              "Tensor, "
              "the output of BeizerAlignOp is a 4-D Tensor with shape "
              "(num_rois, output_channels, pooled_h, pooled_w). "
              "The data type is the same as `x` ");
    AddAttr<float>(
        "sampling_ratio",
        "(float), "
        "the number of channels of the output feature map. "
        "For a task of C classes of objects, output_channels should be "
        "(C + 1) for classification only.");
    AddAttr<float>("spatial_scale",
                   "(float, default 1.0), "
                   "Multiplicative spatial scale factor "
                   "to translate ROI coords from their input scale "
                   "to the scale used when pooling.")
        .SetDefault(1.0);
    AddAttr<int>("pooled_height",
                 "(int, default 1), "
                 "the pooled output height.")
        .SetDefault(1);
    AddAttr<int>("pooled_width",
                 "(int, default 1), "
                 "the pooled output width.")
        .SetDefault(1);
    AddAttr<bool>("aligned",
                  "(bool), "
                  "the pooled output width.")
        .SetDefault(true);
    AddComment(R"Doc(
**BezierAlign Operator,** `rois` **of this op should be a Tensor**
Position sensitive region of interest pooling (also known as BezierAlign) is to perform
position-sensitive average pooling on regions of interest specified by input, takes as 
input N position-sensitive score maps and a list of num_rois regions of interest. 
BezierAlign for ABCNet. Please refer to https://arxiv.org/pdf/2002.10200.pdf for more details.
    )Doc");
  }
};

class BezierAlignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "bezier_align");
    OP_INOUT_CHECK(ctx->HasInput("ROIs"), "Input", "ROIs", "bezier_align");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Input", "Out", "bezier_align");

    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    PADDLE_ENFORCE_EQ(input_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The format of input tensor is NCHW"));

    PADDLE_ENFORCE_EQ(
        rois_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ROIs should be a 2-D LoDTensor of shape (num_rois, 4) "
            "given as [(x1, y1, x2, y2), ...]"));

    int pooled_height = ctx->Attrs().Get<int>("pooled_height");
    int pooled_width = ctx->Attrs().Get<int>("pooled_width");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");
    PADDLE_ENFORCE_GT(pooled_height, 0,
                      platform::errors::InvalidArgument(
                          "The pooled output height must be greater than 0"));
    PADDLE_ENFORCE_GT(pooled_width, 0,
                      platform::errors::InvalidArgument(
                          "The pooled output width must be greater than 0"));
    PADDLE_ENFORCE_GT(spatial_scale, 0.0f,
                      platform::errors::InvalidArgument(
                          "The spatial scale must greater than 0."));

    auto out_dims = input_dims;
    out_dims[0] = rois_dims[0];
    out_dims[1] = input_dims[1];
    out_dims[2] = pooled_height;
    out_dims[3] = pooled_width;

    ctx->SetOutputDim("Out", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bezier_align, ops::BezierAlignOp, ops::BezierAlignOpMaker);
REGISTER_OP_CPU_KERNEL(
    bezier_align,
    ops::CPUBezierAlignOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUBezierAlignOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    bezier_align_grad,
    ops::CPUBezierAlignGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUBezierAlignGradOpKernel<paddle::platform::CPUDeviceContext,
                                    double>);
