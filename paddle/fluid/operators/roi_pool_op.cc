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

#include <memory>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class ROIPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "roi_pool");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "roi_pool");
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
             "(phi::DenseTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D phi::DenseTensor of shape (num_rois, 4)"
             "given as [[x1, y1, x2, y2], ...]. "
             "Where batch_id is the id of the data, "
             "(x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates.");
    AddInput("RoisNum", "(Tensor), The number of RoIs in each image.")
        .AsDispensable();
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
    op->SetInput("RoisNum", this->Input("RoisNum"));
    op->SetInput("Argmax", this->Output("Argmax"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(roi_pool,
                            RoiPoolInferShapeFunctor,
                            PD_INFER_META(phi::RoiPoolInferMeta));

REGISTER_OPERATOR(roi_pool,
                  ops::ROIPoolOp,
                  ops::ROIPoolOpMaker,
                  ops::ROIPoolGradMaker<paddle::framework::OpDesc>,
                  ops::ROIPoolGradMaker<paddle::imperative::OpBase>,
                  RoiPoolInferShapeFunctor);
REGISTER_OPERATOR(roi_pool_grad, ops::ROIPoolGradOp);

REGISTER_OP_VERSION(roi_pool)
    .AddCheckpoint(
        R"ROC(
              Incompatible upgrade of input [RpnRoisLod])ROC",
        paddle::framework::compatible::OpVersionDesc().DeleteInput(
            "RpnRoisLod",
            "Delete RpnRoisLod due to incorrect input name and "
            "it is not used in object detection models yet."))
    .AddCheckpoint(
        R"ROC(
              Upgrade roi_pool add a new input [RoisNum])ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "RoisNum",
            "The number of RoIs in each image. RoisNum is dispensable."));
