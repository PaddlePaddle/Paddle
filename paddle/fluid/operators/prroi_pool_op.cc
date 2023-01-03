/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/prroi_pool_op.h"

#include <memory>

namespace paddle {
namespace operators {

class PRROIPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input of PRROIPoolOp. "
             "The format of input tensor is NCHW. Where N is the batch size, "
             "C is the number of input channels, "
             "H is the height of the input feature map, and "
             "W is the width.");
    AddInput("ROIs",
             "(phi::DenseTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D phi::DenseTensor of shape (num_rois, 4) "
             "given as [(x1, y1, x2, y2), ...]. "
             "where (x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates. "
             "The roi batch index can be calculated from LoD.");
    AddInput("BatchRoINums",
             "(Tensor), "
             "1-D tensor with shape [N], the number of"
             " rois for each image in batch, where N is the batch size")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor), "
              "the output of PRROIPoolOp is a 4-D Tensor with shape "
              "(num_rois, output_channels, pooled_h, pooled_w).");
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
**PRROIPool Operator**

Precise region of interest pooling (also known as PRROIPooling) is to perform
 bilinear interpolation average pooling method for RoI Pooling.

Please refer to https://arxiv.org/abs/1807.11590 for more details.

    )Doc");
  }
};

class PRROIPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "prroi_pool");
    OP_INOUT_CHECK(ctx->HasInput("ROIs"), "Input", "ROIs", "prroi_pool");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Input", "Out", "prroi_pool");

    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    PADDLE_ENFORCE_EQ(input_dims.size(),
                      4,
                      platform::errors::InvalidArgument(
                          "The format of input tensor is NCHW"));
    PADDLE_ENFORCE_EQ(
        rois_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "ROIs should be a 2-D phi::DenseTensor of shape (num_rois, 4) "
            "given as [(x1, y1, x2, y2), ...]"));
    PADDLE_ENFORCE_EQ(
        rois_dims[1],
        4,
        platform::errors::InvalidArgument(
            "ROIs should be a 2-D phi::DenseTensor of shape (num_rois, 4) "
            "given as [(x1, y1, x2, y2), ...]"));
    int pooled_height = ctx->Attrs().Get<int>("pooled_height");
    int pooled_width = ctx->Attrs().Get<int>("pooled_width");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");

    PADDLE_ENFORCE_GT(pooled_height,
                      0,
                      platform::errors::InvalidArgument(
                          "The pooled output height must be greater than 0"));
    PADDLE_ENFORCE_GT(pooled_width,
                      0,
                      platform::errors::InvalidArgument(
                          "The pooled output width must be greater than 0"));
    PADDLE_ENFORCE_GT(spatial_scale,
                      0.0f,
                      platform::errors::InvalidArgument(
                          "The spatial scale must greater than 0."));

    auto out_dims = input_dims;
    out_dims[0] = rois_dims[0];
    out_dims[1] = input_dims[1];
    out_dims[2] = pooled_height;
    out_dims[3] = pooled_width;

    if (ctx->HasInput("BatchRoINums")) {
      auto rois_batch_index = ctx->GetInputDim("BatchRoINums");
      PADDLE_ENFORCE_EQ(rois_batch_index[0],
                        input_dims[0],
                        platform::errors::InvalidArgument(
                            "The length of BatchRoINums should equal to  "
                            "first dim of inputs(X)"));
    }
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

class PRROIPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "prroi_pool");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "prroi_pool");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->SetOutputDim(framework::GradVarName("ROIs"), ctx->GetInputDim("ROIs"));
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
class PRROIPoolGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("prroi_pool_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("BatchRoINums", this->Input("BatchRoINums"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("ROIs"), this->InputGrad("ROIs"));
    op->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(prroi_pool,
                  ops::PRROIPoolOp,
                  ops::PRROIPoolOpMaker,
                  ops::PRROIPoolGradMaker<paddle::framework::OpDesc>,
                  ops::PRROIPoolGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(prroi_pool_grad, ops::PRROIPoolGradOp);
REGISTER_OP_CPU_KERNEL(prroi_pool,
                       ops::CPUPRROIPoolOpKernel<phi::CPUContext, float>,
                       ops::CPUPRROIPoolOpKernel<phi::CPUContext, double>,
                       ops::CPUPRROIPoolOpKernel<phi::CPUContext, int>,
                       ops::CPUPRROIPoolOpKernel<phi::CPUContext, int64_t>);
REGISTER_OP_CPU_KERNEL(prroi_pool_grad,
                       ops::CPUPRROIPoolGradOpKernel<phi::CPUContext, float>,
                       ops::CPUPRROIPoolGradOpKernel<phi::CPUContext, double>,
                       ops::CPUPRROIPoolGradOpKernel<phi::CPUContext, int>,
                       ops::CPUPRROIPoolGradOpKernel<phi::CPUContext, int64_t>);
