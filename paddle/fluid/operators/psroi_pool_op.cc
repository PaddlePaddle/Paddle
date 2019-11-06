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

#include "paddle/fluid/operators/psroi_pool_op.h"
#include <memory>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class PSROIPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, "
             "the input of PSROIPoolOp. "
             "The format of input tensor is NCHW. Where N is the batch size, "
             "C is the number of input channels, "
             "H is the height of the input feature map, and "
             "W is the width. The data type can be float32 or float64");
    AddInput("ROIs",
             "LoDTensor, "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D LoDTensor of shape (num_rois, 4) "
             "given as [(x1, y1, x2, y2), ...]. "
             "where (x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates. "
             "The roi batch index can be calculated from LoD.");
    AddOutput("Out",
              "Tensor, "
              "the output of PSROIPoolOp is a 4-D Tensor with shape "
              "(num_rois, output_channels, pooled_h, pooled_w). "
              "The data type is the same as `x` ");
    AddAttr<int>(
        "output_channels",
        "(int), "
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
    AddComment(R"Doc(
**PSROIPool Operator,** `rois` **of this op should be a LoDTensor**

Position sensitive region of interest pooling (also known as PSROIPooling) is to perform
position-sensitive average pooling on regions of interest specified by input, takes as 
input N position-sensitive score maps and a list of num_rois regions of interest. 

PSROIPooling for R-FCN. Please refer to https://arxiv.org/abs/1605.06409 for more details.
    )Doc");
  }
};

class PSROIPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of PSROIPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("ROIs"),
                   "Input(ROIs) of PSROIPoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of PSROIPoolOp should not be null.");
    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    PADDLE_ENFORCE(input_dims.size() == 4,
                   "The format of input tensor is NCHW");
    PADDLE_ENFORCE(rois_dims.size() == 2,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 4) "
                   "given as [(x1, y1, x2, y2), ...]");
    PADDLE_ENFORCE(rois_dims[1] == 4,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 4) "
                   "given as [(x1, y1, x2, y2), ...]");

    int pooled_height = ctx->Attrs().Get<int>("pooled_height");
    int pooled_width = ctx->Attrs().Get<int>("pooled_width");
    int output_channels = ctx->Attrs().Get<int>("output_channels");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");

    PADDLE_ENFORCE(
        input_dims[1] == output_channels * pooled_height * pooled_width,
        "the channel of X(%d) should be equal to the product of "
        "output_channels(%d), pooled_height(%d) and pooled_width(%d)",
        input_dims[1], output_channels, pooled_height, pooled_width);

    PADDLE_ENFORCE_GT(pooled_height, 0,
                      "The pooled output height must be greater than 0");
    PADDLE_ENFORCE_GT(pooled_width, 0,
                      "The pooled output width must be greater than 0");
    PADDLE_ENFORCE_GT(output_channels, 1,
                      "The pooled output channels must greater than 1");
    PADDLE_ENFORCE_GT(spatial_scale, 0.0f,
                      "The spatial scale must greater than 0.");

    auto out_dims = input_dims;
    out_dims[0] = rois_dims[0];
    out_dims[1] =
        output_channels;  // input_dims[1] / (pooled_height * pooled_width);
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

class PSROIPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "The gradient of X should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

template <typename T>
class PSROIPoolGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("psroi_pool_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(psroi_pool, ops::PSROIPoolOp, ops::PSROIPoolOpMaker,
                  ops::PSROIPoolGradMaker<paddle::framework::OpDesc>,
                  ops::PSROIPoolGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(psroi_pool_grad, ops::PSROIPoolGradOp);
REGISTER_OP_CPU_KERNEL(
    psroi_pool,
    ops::CPUPSROIPoolOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUPSROIPoolOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    psroi_pool_grad,
    ops::CPUPSROIPoolGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUPSROIPoolGradOpKernel<paddle::platform::CPUDeviceContext, double>);
