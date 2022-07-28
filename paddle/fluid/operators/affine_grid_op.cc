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
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;


class AffineGridOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library{framework::LibraryType::kPlain};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::CanCUDNNBeUsed(ctx)) {
      library = framework::LibraryType::kCUDNN;
    }
#endif
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Theta");
    return framework::OpKernelType(
        data_type, ctx.GetPlace(), framework::DataLayout::kAnyLayout, library);
  }
};

class AffineGridOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Theta",
        "(Tensor) A batch of affine transform parameters with shape [N, 2, 3]. "
        "It is used to transform coordinate (x_0, y_0) to coordinate (x_1, "
        "y_1).");
    AddInput("OutputShape",
             "(Tensor) The shape of target image with format [N, C, H, W].")
        .AsDispensable();
    AddOutput("Output", "(Tensor) Output Tensor with shape [N, H, W, 2].");
    AddAttr<bool>(
        "use_cudnn",
        "(bool, default false) Only used in cudnn kernel, need install cudnn")
        .SetDefault(true)
        .AsExtra();
    AddAttr<bool>("align_corners",
                  "(bool, default false) Whether to align the corners of input"
                  "and output.")
        .SetDefault(true);
    AddAttr<std::vector<int>>(
        "output_shape",
        "The target output image shape with format [N, C, H, W].")
        .SetDefault(std::vector<int>());

    AddComment(R"DOC(
    It generates a grid of (x,y) coordinates using the parameters of the
    affine transformation that correspond to a set of points where the input
    feature map should be sampled to produce the transformed output feature map.

    Given:
        Theta = [[[x_11, x_12, x_13]
                  [x_14, x_15, x_16]]
                 [[x_21, x_22, x_23]
                  [x_24, x_25, x_26]]]
    
        OutputShape = [2, 3, 5, 5]

    Step 1:

        Generate relative coordinates according to OutputShape.
        The values of relative coordinates are in the interval between -1 and 1.
        The shape of the relative coordinates is [2, H, W] as below:
    
        C = [[[-1.  -1.  -1.  -1.  -1. ]
              [-0.5 -0.5 -0.5 -0.5 -0.5]
              [ 0.   0.   0.   0.   0. ]
              [ 0.5  0.5  0.5  0.5  0.5]
              [ 1.   1.   1.   1.   1. ]] 
             [[-1.  -0.5  0.   0.5  1. ]
              [-1.  -0.5  0.   0.5  1. ]
              [-1.  -0.5  0.   0.5  1. ]
              [-1.  -0.5  0.   0.5  1. ]
              [-1.  -0.5  0.   0.5  1. ]]]
        C[0] is the coordinates in height axis and  C[1] is the coordinates in
        width axis.
    
    Step2:
        Tanspose and reshape C to shape [H * W, 2] and append ones to last
        dimension. The we get:
        C_ = [[-1.  -1.   1. ]
              [-0.5 -1.   1. ]
              [ 0.  -1.   1. ]
              [ 0.5 -1.   1. ]
              [ 1.  -1.   1. ]
              [-1.  -0.5  1. ]
              [-0.5 -0.5  1. ]
              [ 0.  -0.5  1. ]
              [ 0.5 -0.5  1. ]
              [ 1.  -0.5  1. ]
              [-1.   0.   1. ]
              [-0.5  0.   1. ]
              [ 0.   0.   1. ]
              [ 0.5  0.   1. ]
              [ 1.   0.   1. ]
              [-1.   0.5  1. ]
              [-0.5  0.5  1. ]
              [ 0.   0.5  1. ]
              [ 0.5  0.5  1. ]
              [ 1.   0.5  1. ]
              [-1.   1.   1. ]
              [-0.5  1.   1. ]
              [ 0.   1.   1. ]
              [ 0.5  1.   1. ]
              [ 1.   1.   1. ]]
    Step3:
        Compute output by equation $$Output[i] = C_ * Theta[i]^T$$
    )DOC");
  }
};

class AffineGridOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::CanCUDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kCUDNN;
    }
#endif
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Output")),
                                   ctx.GetPlace(),
                                   framework::DataLayout::kAnyLayout,
                                   library_);
  }
};

template <typename T>
class AffineGridGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("affine_grid_grad");
    op->SetInput("OutputShape", this->Input("OutputShape"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("Theta"), this->InputGrad("Theta"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(affine_grid, AffineGridInferMetaFunctor,
                            PD_INFER_META(phi::AffineGridInferMeta));
REGISTER_OPERATOR(affine_grid,
                  ops::AffineGridOp,
                  ops::AffineGridOpMaker,
                  ops::AffineGridGradMaker<paddle::framework::OpDesc>,
                  ops::AffineGridGradMaker<paddle::imperative::OpBase>,
                  AffineGridInferMetaFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(affine_grid_grad, AffineGridGradInferMetaFunctor,
                            PD_INFER_META(phi::AffineGridGradInferMeta));
REGISTER_OPERATOR(affine_grid_grad,
                  ops::AffineGridOpGrad,
                  AffineGridGradInferMetaFunctor);

REGISTER_OP_VERSION(affine_grid)
    .AddCheckpoint(
        R"ROC(
               Compatible upgrade of affine_grid, add a new attribute [align_corners])ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "align_corners",
            "Whether to align the corners of input and output.",
            true));
