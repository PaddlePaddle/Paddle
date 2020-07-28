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

#include "paddle/fluid/operators/roi_pool_op.h"
#include <memory>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class ROIPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "roi_pool");
    OP_INOUT_CHECK(ctx->HasInput("ROIs"), "Input", "ROIs", "roi_pool");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "roi_pool");
    OP_INOUT_CHECK(ctx->HasOutput("Argmax"), "Output", "Argmax", "roi_pool");

    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    if (ctx->HasInput("RoisLod")) {
      auto rois_lod_dims = ctx->GetInputDim("RoisLod");
      PADDLE_ENFORCE_EQ(rois_lod_dims.size(), 1,
                        platform::errors::InvalidArgument(
                            "The lod information tensor of ROIs should "
                            "be one-dimensional"));
    }
    PADDLE_ENFORCE_EQ(input_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input data should be a four-dimensional "
                          "tensor with [N,C,H,W], but received input data with "
                          " %d dimension",
                          input_dims.size()));
    PADDLE_ENFORCE_EQ(
        rois_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ROIs should be a 2-D LoDTensor with shape (num_rois, 4)"
            "given as [[x1, y1, x2, y2], ...], but received ROIs is "
            "%d-dimensional LoDTensor",
            rois_dims.size()));
    PADDLE_ENFORCE_EQ(
        rois_dims[1], kROISize,
        platform::errors::InvalidArgument(
            "ROIs should be a 2-D LoDTensor with shape (num_rois, 4)"
            "given as [[x1, y1, x2, y2], ...]. But the second dimension of  "
            "the received data is %d",
            rois_dims[1]));

    int pooled_height = ctx->Attrs().Get<int>("pooled_height");
    int pooled_width = ctx->Attrs().Get<int>("pooled_width");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");

    PADDLE_ENFORCE_GT(pooled_height, 0,
                      platform::errors::OutOfRange(
                          "The pooled output height must be greater than 0"
                          "but received height is %d",
                          pooled_height));
    PADDLE_ENFORCE_GT(pooled_width, 0,
                      platform::errors::OutOfRange(
                          "The pooled output width must be greater than 0"
                          "but received width is %d",
                          pooled_width));
    PADDLE_ENFORCE_GT(spatial_scale, 0.0f,
                      platform::errors::OutOfRange(
                          "The spatial scale must be greater than 0, "
                          "but received spatial scale is %f",
                          spatial_scale));

    auto out_dims = input_dims;
    out_dims[0] = rois_dims[0];
    out_dims[1] = input_dims[1];
    out_dims[2] = pooled_height;
    out_dims[3] = pooled_width;

    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("Argmax", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class ROIPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "roi_pool");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "roi_pool");
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class ROIPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input of ROIPoolOp. "
             "The format of input tensor is NCHW. Where N is batch size, "
             "C is the number of input channels, "
             "H is the height of the feature, and "
             "W is the width of the feature.");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D LoDTensor of shape (num_rois, 4)"
             "given as [[x1, y1, x2, y2], ...]. "
             "Where batch_id is the id of the data, "
             "(x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates.");
    AddInput("RoisLod", "(Tensor), The lod info of rois.").AsDispensable();
    AddOutput("Out",
              "(Tensor), "
              "The output of ROIPoolOp is a 4-D tensor with shape "
              "(num_rois, channels, pooled_h, pooled_w).");
    AddOutput("Argmax",
              "(Tensor), "
              "Argmaxes corresponding to indices in X used "
              "for gradient computation. Only output "
              "if arg \"is_test\" is false.")
        .AsIntermediate();
    AddAttr<float>("spatial_scale",
                   "(float, default 1.0), "
                   "Multiplicative spatial scale factor "
                   "to translate ROI coords from their input scale "
                   "to the scale used when pooling.")
        .SetDefault(1.0);
    AddAttr<int>("pooled_height",
                 "(int, default 1), "
                 "The pooled output height.")
        .SetDefault(1);
    AddAttr<int>("pooled_width",
                 "(int, default 1), "
                 "The pooled output width.")
        .SetDefault(1);
    AddComment(R"DOC(
**ROIPool Operator**

Region of interest pooling (also known as RoI pooling) is to perform
is to perform max pooling on inputs of nonuniform sizes to obtain
fixed-size feature maps (e.g. 7*7).

The operator has three steps:

1. Dividing each region proposal into equal-sized sections with
   the pooled_width and pooled_height

2. Finding the largest value in each section

3. Copying these max values to the output buffer

ROI Pooling for Faster-RCNN. The link below is a further introduction: 
https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn
    )DOC");
  }
};

template <typename T>
class ROIPoolGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("roi_pool_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("RoisLod", this->Input("RoisLod"));
    op->SetInput("Argmax", this->Output("Argmax"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(roi_pool, ops::ROIPoolOp, ops::ROIPoolOpMaker,
                  ops::ROIPoolGradMaker<paddle::framework::OpDesc>,
                  ops::ROIPoolGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(roi_pool_grad, ops::ROIPoolGradOp);
REGISTER_OP_CPU_KERNEL(
    roi_pool,
    ops::CPUROIPoolOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUROIPoolOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CPUROIPoolOpKernel<paddle::platform::CPUDeviceContext, int>);
REGISTER_OP_CPU_KERNEL(
    roi_pool_grad,
    ops::CPUROIPoolGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUROIPoolGradOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CPUROIPoolGradOpKernel<paddle::platform::CPUDeviceContext, int>);
