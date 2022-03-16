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

#include <memory>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class ROIAlignOp : public framework::OperatorWithKernel {
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

class ROIAlignGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::NotFound("The GRAD@Out of ROIAlignGradOp "
                                   "is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasOutputs(framework::GradVarName("X")), true,
                      platform::errors::NotFound("The GRAD@X of ROIAlignGradOp "
                                                 "is not found."));
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "ROIs"),
        ctx.device_context());
  }
};

class ROIAlignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "The input of ROIAlignOp. The data type is float32 or float64."
             "The format of input tensor is NCHW. Where N is batch size, "
             "C is the number of input channels, "
             "H is the height of the feature, and "
             "W is the width of the feature.");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D LoDTensor of shape (num_rois, 4)"
             "given as [[x1, y1, x2, y2], ...]. "
             "(x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates.");
    AddInput("RoisNum",
             "(Tensor), "
             "The number of RoIs in each image.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor), "
              "The output of ROIAlignOp is a 4-D tensor with shape "
              "(num_rois, channels, pooled_h, pooled_w). The data type is "
              "float32 or float64.");
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
    AddAttr<int>("sampling_ratio",
                 "(int,default -1),"
                 "number of sampling points in the interpolation grid"
                 "If <=0, then grid points are adaptive to roi_width "
                 "and pooled_w, likewise for height")
        .SetDefault(-1);
    AddAttr<bool>("aligned",
                  "(bool, default False),"
                  "If true, pixel shift it by -0.5 for align more perfectly")
        .SetDefault(false);
    AddComment(R"DOC(
**RoIAlign Operator**

Region of interest align (also known as RoI align) is to perform
bilinear interpolation on inputs of nonuniform sizes to obtain 
fixed-size feature maps (e.g. 7*7)

Dividing each region proposal into equal-sized sections with
the pooled_width and pooled_height. Location remains the origin
result.

In each ROI bin, the value of the four regularly sampled locations 
are computed directly through bilinear interpolation. The output is
the mean of four locations.
Thus avoid the misaligned problem.   
    )DOC");
  }
};

template <typename T>
class ROIAlignGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("roi_align_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("RoisNum", this->Input("RoisNum"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(RoiAlignGradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(roi_align, RoiAlignInferShapeFunctor,
                            PD_INFER_META(phi::RoiAlignInferMeta));

REGISTER_OPERATOR(roi_align, ops::ROIAlignOp, ops::ROIAlignOpMaker,
                  ops::ROIAlignGradMaker<paddle::framework::OpDesc>,
                  ops::ROIAlignGradMaker<paddle::imperative::OpBase>,
                  RoiAlignInferShapeFunctor);
REGISTER_OPERATOR(roi_align_grad, ops::ROIAlignGradOp,
                  ops::RoiAlignGradNoNeedBufVarsInferer);

REGISTER_OP_VERSION(roi_align)
    .AddCheckpoint(
        R"ROC(
              Incompatible upgrade of input [RpnRoisLod])ROC",
        paddle::framework::compatible::OpVersionDesc().DeleteInput(
            "RpnRoisLod",
            "Delete RpnRoisLod due to incorrect input name and "
            "it is not used in object detection models yet."))
    .AddCheckpoint(
        R"ROC(
             Upgrade roi_align add a new input [RoisNum])ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "RoisNum",
            "The number of RoIs in each image. RoisNum is dispensable."))
    .AddCheckpoint(
        R"ROC(
             Upgrade roi_align add a new input [aligned])ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "aligned",
            "If true, pixel shift it by -0.5 for align more perfectly.",
            false));
