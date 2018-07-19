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

#include <string>
#include "paddle/fluid/framework/op_registry.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class AffineGridOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Theta"),
                   "Input(Theta) of AffineGridOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Size"),
                   "Input(Size) of AffineGridOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Output"),
                   "Output(Output) of AffineGridOp should not be null.");
    auto theta_dims = ctx->GetInputDim("Theta");
    auto size_dims = ctx->GetInputDim("Size");
    PADDLE_ENFORCE(theta_dims.size() == 3,
                   "AffineGrid's Input(Theta) should be 3-D tensor.");
    PADDLE_ENFORCE(size_dims.size() == 1,
                   "AffineGrid's Input(Size) should be 1-D tensor.");
    PADDLE_ENFORCE(theta_dims[1] == 2, "Input(theta) dims[1] should be 2.");
    PADDLE_ENFORCE(theta_dims[2] == 3, "Input(theta) dims[2] should be 3.");
    // N * H * W * 2
    ctx->SetOutputDim("Output",
                      framework::make_ddim({theta_dims[0], -1, -1, 2}));
    ctx->ShareLoD("Theta", "Output");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_CUDA
    if (platform::CanCUDNNBeUsed(ctx)) {
      library = framework::LibraryType::kCUDNN;
    }
#endif
    auto data_type = framework::ToDataType(ctx.Input<Tensor>("Theta")->type());
    return framework::OpKernelType(data_type, ctx.GetPlace(),
                                   framework::DataLayout::kAnyLayout, library);
  }
};

class AffineGridOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Theta", "(Tensor) Input batch of affine matrices (N×2×3).");
    AddInput("Size", "(Tensor) the target output image size (N×C×H×W).");
    AddOutput("Output", "(Tensor) output Tensor of size (N×H×W×2).");
    AddAttr<bool>(
        "use_cudnn",
        "(bool, default false) Only used in cudnn kernel, need install cudnn")
        .SetDefault(true);
    AddComment(R"DOC(
    )DOC");
  }
};

class AffineGridOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    auto theta_dims = ctx->GetInputDim("Theta");
    auto size_dims = ctx->GetInputDim("Size");
    if (ctx->HasOutput(framework::GradVarName("Theta"))) {
      ctx->SetOutputDim(framework::GradVarName("Theta"), theta_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_CUDA
    if (platform::CanCUDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kCUDNN;
    }
#endif
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Theta")->type()),
        ctx.GetPlace(), framework::DataLayout::kAnyLayout, library_);
  }
};

class AffineGridGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("affine_grid_grad");
    op->SetInput("Theta", Input("Theta"));
    op->SetInput("Size", Input("Size"));
    op->SetInput(framework::GradVarName("Output"), OutputGrad("Output"));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("Theta"), InputGrad("Theta"));
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(affine_grid, ops::AffineGridOp, ops::AffineGridOpMaker,
                  ops::AffineGridGradMaker);
REGISTER_OPERATOR(affine_grid_grad, ops::AffineGridOpGrad);
