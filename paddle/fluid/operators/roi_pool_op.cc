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
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ROIPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("ROIs"),
                   "Input(ROIs) of ROIPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ROIPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Argmax"),
                   "Output(Argmax) of ROIPoolOp should not be null.");
    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    PADDLE_ENFORCE(input_dims.size() == 4,
                   "The format of input tensor is NCHW.");
    PADDLE_ENFORCE(rois_dims.size() == 2,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 4)"
                   "given as [[x1, y1, x2, y2], ...].");
    PADDLE_ENFORCE(rois_dims[1] == kROISize,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 4)"
                   "given as [[x1, y1, x2, y2], ...].");

    int pooled_height = ctx->Attrs().Get<int>("pooled_height");
    int pooled_width = ctx->Attrs().Get<int>("pooled_width");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");

    PADDLE_ENFORCE_GT(pooled_height, 0,
                      "The pooled output height must greater than 0");
    PADDLE_ENFORCE_GT(pooled_width, 0,
                      "The pooled output width must greater than 0");
    PADDLE_ENFORCE_GT(spatial_scale, 0.0f,
                      "The spatial scale must greater than 0");

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
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName("X")),
                   "The gradient of X should not be null.");
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
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("roi_pool_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("Argmax", this->Output("Argmax"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
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
    ops::CPUROIPoolOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    roi_pool_grad,
    ops::CPUROIPoolGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUROIPoolGradOpKernel<paddle::platform::CPUDeviceContext, double>);
