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

#include "paddle/fluid/operators/affine_grid_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct Linspace<paddle::platform::CPUDeviceContext, T> {
  void operator()(T start, T end, int count, bool align_corners,
                  framework::Tensor* numbers,
                  const framework::ExecutionContext& ctx) {
    T* number_data = numbers->mutable_data<T>({count}, platform::CPUPlace());
    T slice = (end - start) / (T)(count - 1);
    if (!align_corners) {
      slice = (end - start) / (T)count;
      start *= (T)(count - 1) / (T)count;
    }
    for (int i = 0; i < count; ++i) {
      number_data[i] = start + (T)i * slice;
    }
  }
};

class AffineGridOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Theta"), true,
                      platform::errors::NotFound(
                          "The input 'Theta' of AffineGridOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Output"), true,
                      platform::errors::NotFound(
                          "The output 'Output' of AffineGridOp is not found."));
    auto theta_dims = ctx->GetInputDim("Theta");
    PADDLE_ENFORCE_EQ(
        theta_dims.size(), 3,
        platform::errors::InvalidArgument(
            "The input Theta's dimensions size should be 3. But received "
            "Theta's demensions size=[%d],  Theta's dimensions=[%s].",
            theta_dims.size(), theta_dims));

    auto output_shape = ctx->Attrs().Get<std::vector<int>>("output_shape");
    if (output_shape.size() == 0) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("OutputShape"), true,
          platform::errors::NotFound(
              "The input 'OutputShape' of AffineGridOp should not be null if "
              "'output_shape' is not configured."));
      auto output_shape_dims = ctx->GetInputDim("OutputShape");
      PADDLE_ENFORCE_EQ(
          output_shape_dims.size(), 1,
          platform::errors::InvalidArgument(
              "The dimesions size of input OutputShape in AffineGridOp should "
              "be 1. But received OutputShape's  dimesions size=[%d], "
              "OutputShape's  dimesions=[%s]",
              output_shape_dims.size(), output_shape_dims));
    } else {
      PADDLE_ENFORCE_EQ(
          output_shape.size(), 4,
          platform::errors::InvalidArgument(
              "The size of attribute 'output_shape' in AffineGridOp should be "
              "4. But received output_shape's size=[%d].",
              output_shape.size()));
    }

    PADDLE_ENFORCE_EQ(
        theta_dims[1], 2,
        platform::errors::InvalidArgument(
            "The second dimesion of input 'theta' in AffineGridOp should be 2. "
            "But received second dimesion=[%d], dimesions=[%s]",
            theta_dims[1], theta_dims));
    PADDLE_ENFORCE_EQ(
        theta_dims[2], 3,
        platform::errors::InvalidArgument(
            "The third dimesion of input 'theta' in AffineGridOp should be 3. "
            "But received third dimesion=[%d], dimesions=[%s]",
            theta_dims[2], theta_dims));

    // N * H * W * 2
    ctx->SetOutputDim("Output",
                      framework::make_ddim({theta_dims[0], -1, -1, 2}));
    ctx->ShareLoD("Theta", "Output");
  }

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
    return framework::OpKernelType(data_type, ctx.GetPlace(),
                                   framework::DataLayout::kAnyLayout, library);
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
        .SetDefault(true);
    AddAttr<bool>("align_corners",
                  "(bool, default false) Whether to align the corners of input"
                  "and ouput.")
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
  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("Theta"))) {
      auto output_dims = ctx->GetInputDim(framework::GradVarName("Output"));
      ctx->SetOutputDim(framework::GradVarName("Theta"),
                        {output_dims[0], 2, 3});
    }
  }

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
                                   framework::DataLayout::kAnyLayout, library_);
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
REGISTER_OPERATOR(affine_grid, ops::AffineGridOp, ops::AffineGridOpMaker,
                  ops::AffineGridGradMaker<paddle::framework::OpDesc>,
                  ops::AffineGridGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(affine_grid_grad, ops::AffineGridOpGrad);

REGISTER_OP_CPU_KERNEL(
    affine_grid,
    ops::AffineGridOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AffineGridOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    affine_grid_grad,
    ops::AffineGridGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AffineGridGradOpKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_VERSION(affine_grid)
    .AddCheckpoint(
        R"ROC(
               Compatible upgrade of affine_grid, add a new attribute [align_corners])ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "align_corners",
            "Whether to align the corners of input and output.", true));
