/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/roi_perspective_transform_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class ROIPerspectiveTransformOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ROIPerspectiveTransformOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("ROIs"),
        "Input(ROIs) of ROIPerspectiveTransformOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of ROIPerspectiveTransformOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Mask"),
        "Output(Mask) of ROIPerspectiveTransformOp should not be null.");
    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    PADDLE_ENFORCE(input_dims.size() == 4,
                   "The format of input tensor is NCHW.");
    PADDLE_ENFORCE(rois_dims.size() == 9,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 8)"
                   "given as [[x0, y0, x1, y1, x2, y2, x3, y3], ...]");
    PADDLE_ENFORCE(rois_dims[1] == 8,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 8)"
                   "given as [[x0, y0, x1, y1, x2, y2, x3, y3], ...].");

    int transformed_height = ctx->Attrs().Get<int>("transformed_height");
    int transformed_width = ctx->Attrs().Get<int>("transformed_width");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");

    PADDLE_ENFORCE_GT(transformed_height, 0,
                      "The transformed output height must greater than 0");
    PADDLE_ENFORCE_GT(transformed_width, 0,
                      "The transformed output width must greater than 0");
    PADDLE_ENFORCE_GT(spatial_scale, 0.0f,
                      "The spatial scale must greater than 0");

    auto out_dims = framework::DDim({rois_dims[0],   // num_rois
                                     input_dims[1],  // channels
                                     transformed_height, transformed_widt});

    auto mask_dims = framework::DDim({rois_dims[0],  // num_rois
                                      1,             // channels
                                      transformed_height, transformed_widt});

    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("Mask", mask_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
  }
};

class ROIPerspectiveTransformGradOp : public framework::OperatorWithKernel {
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
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
  }
};

class ROIPerspectiveTransformOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input of ROIPerspectiveTransformOp. "
             "The format of input tensor is NCHW. Where N is batch size, "
             "C is the number of input channels, "
             "H is the height of the feature, and "
             "W is the width of the feature.");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D LoDTensor of shape (num_rois, 4)"
             "given as [[x1, y1, x2, y2], "
             "Where batch_id is the id of the data, "
             "(x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates.");
    AddOutput(
        "Out",
        "(Tensor), "
        "The output of ROIPerspectiveTransformOp is a 4-D tensor with shape "
        "(num_rois, channels, pooled_h, pooled_w).");
    AddOutput("mask",
              "(Tensor), "
              "");
    AddAttr<float>("spatial_scale",
                   "(float, default 1.0), "
                   "Multiplicative spatial scale factor "
                   "to translate ROI coords from their input scale "
                   "to the scale used when pooling.")
        .SetDefault(1.0);
    AddAttr<int>("transformed_height",
                 "(int, default 1), "
                 "The transformed_height output height.")
        .SetDefault(1);
    AddAttr<int>("transformed_width",
                 "(int, default 1), "
                 "The transformed output width.")
        .SetDefault(1);
    AddComment(R"DOC(
**ROIPerspectiveTransform Operator**

    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(roi_perspective_transform, ops::ROIPerspectiveTransformOp,
                  ops::ROIPerspectiveTransformOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(roi_perspective_transform_grad,
                  ops::ROIPerspectiveTransformGradOp);
REGISTER_OP_CPU_KERNEL(
    roi_perspective_transform,
    ops::CPUROIPerspectiveTransformOpKernel<paddle::platform::CPUDeviceContext,
                                            float>,
    ops::CPUROIPerspectiveTransformOpKernel<paddle::platform::CPUDeviceContext,
                                            double>);
REGISTER_OP_CPU_KERNEL(roi_perspective_transform_grad,
                       ops::CPUROIPerspectiveTransformGradOpKernel<
                           paddle::platform::CPUDeviceContext, float>,
                       ops::CPUROIPerspectiveTransformOpKernel<
                           paddle::platform::CPUDeviceContext, double>);
