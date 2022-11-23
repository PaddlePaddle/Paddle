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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class PSROIPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input of PSROIPoolOp. "
             "The format of input tensor is NCHW. Where N is the batch size, "
             "C is the number of input channels, "
             "H is the height of the input feature map, and "
             "W is the width. The data type can be float32 or float64");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D LoDTensor of shape (num_rois, 4) "
             "given as [(x1, y1, x2, y2), ...]. "
             "where (x1, y1) is the top left coordinates, and "
             "(x2, y2) is the bottom right coordinates. "
             "The roi batch index can be calculated from LoD.");
    AddInput("RoisNum",
             "(Tensor), "
             "The number of RoIs in each image.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor), "
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
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("psroi_pool_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("RoisNum", this->Input("RoisNum"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(psroi_pool,
                            PsroiPoolInferShapeFunctor,
                            PD_INFER_META(phi::PsroiPoolInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(psroi_pool_grad,
                            PsroiPoolGradInferShapeFunctor,
                            PD_INFER_META(phi::PsroiPoolGradInferMeta));
REGISTER_OPERATOR(psroi_pool,
                  ops::PSROIPoolOp,
                  ops::PSROIPoolOpMaker,
                  ops::PSROIPoolGradMaker<paddle::framework::OpDesc>,
                  ops::PSROIPoolGradMaker<paddle::imperative::OpBase>,
                  PsroiPoolInferShapeFunctor);
REGISTER_OPERATOR(psroi_pool_grad,
                  ops::PSROIPoolGradOp,
                  PsroiPoolGradInferShapeFunctor);
