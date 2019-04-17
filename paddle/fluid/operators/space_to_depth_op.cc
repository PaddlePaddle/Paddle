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

#include "paddle/fluid/operators/space_to_depth_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class SpaceToDepthOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SpaceToDepthOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SpaceToDepthOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 4, "input should be a 4D tensor");
    auto blocksize = ctx->Attrs().Get<int64_t>("blocksize");

    PADDLE_ENFORCE_GT(blocksize, 1, "The blocksize should be Greater than 1");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_GT(x_dims[1], 0, "input channel should be Greater than 0");
      PADDLE_ENFORCE_GT(x_dims[2], 0, "input Height should be Greater than 0");
      PADDLE_ENFORCE_GT(x_dims[3], 0, "input Width should be Greater than 0");

      PADDLE_ENFORCE_EQ(x_dims[1] % (blocksize * blocksize), 0,
                        "input channel should be divisible of the square of "
                        "SpaceToDepthOp blocksize");
      PADDLE_ENFORCE_EQ(x_dims[2] % (blocksize), 0,
                        "input Height should be divisible of the square of "
                        "SpaceToDepthOp blocksize");
      PADDLE_ENFORCE_EQ(x_dims[3] % (blocksize), 0,
                        "input Width should be divisible of the square of "
                        "SpaceToDepthOp blocksize");
    } else {
      if (x_dims[1] != -1) {
        PADDLE_ENFORCE_GT(x_dims[1], 0,
                          "input channel should be Greater than 0");
        PADDLE_ENFORCE_EQ(x_dims[1] % (blocksize * blocksize), 0,
                          "input channel should be divisible of the square of "
                          "SpaceToDepthOp blocksize");
      }
      if (x_dims[2] != -1) {
        PADDLE_ENFORCE_GT(x_dims[2], 0,
                          "input Height should be Greater than 0");
        PADDLE_ENFORCE_EQ(x_dims[2] % (blocksize), 0,
                          "input Height should be divisible of the square of "
                          "SpaceToDepthOp blocksize");
      }

      if (x_dims[3] != -1) {
        PADDLE_ENFORCE_GT(x_dims[3], 0, "input Width should be Greater than 0");

        PADDLE_ENFORCE_EQ(x_dims[3] % (blocksize), 0,
                          "input Width should be divisible of the square of "
                          "SpaceToDepthOp blocksize");
      }
    }

    VLOG(3) << "SpaceToDepthOp operator x.shape=" << x_dims
            << "Attribute blocksize" << blocksize << std::endl;

    std::vector<int64_t> output_shape(4, 0);  // [B,C,H,W]
    output_shape[0] = x_dims[0];
    output_shape[1] = x_dims[1] * blocksize * blocksize;
    output_shape[2] = x_dims[2] / blocksize;
    output_shape[3] = x_dims[3] / blocksize;

    auto out_dims = framework::make_ddim(output_shape);

    ctx->SetOutputDim("Out", out_dims);

    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class SpaceToDepthOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor). The input should be a 4D tensor B * C * W * H of "
             "SpaceToDepthOp "
             "operator.");
    AddOutput("Out",
              "(Tensor), The output should be a 4D tensor B * C2 * W2 * H2 of "
              "SpaceToDepthOp operator.");
    AddAttr<int64_t>(
        "blocksize",
        "(int64_t, default 2) blocksize used to do change Space To Depth.")
        .SetDefault(2)
        .GreaterThan(1);
    AddComment(R"DOC(
        reorg operator used in Yolo v2.
        The equation is: C2 = C1/blocksize * blocksize, W2 = W1 * blocksize + offset % blocksize, H2 = H1 * blocksize + offset / blocksize,

        Reshape Input(X) into the shape according to Attr(blocksize). The
        data in Input(X) are unchanged.

        Examples:

            1. Given a 4-D tensor Input(X) with a shape [128, 2048, 26, 26], and the blocksize is 2, the reorg operator will transform Input(X)
            into a 4-D tensor with shape [128, 2048, 13, 13] and leaving Input(X)'s data unchanged.

    )DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(SpaceToDepthGradOpNoBuffer, "X");

class SpaceToDepthGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());

    op->SetType("space_to_depth_grad");

    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetInput("X", Input("X"));

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

class SpaceToDepthGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(space_to_depth, ops::SpaceToDepthOp, ops::SpaceToDepthOpMaker,
                  ops::SpaceToDepthGradOpDescMaker);
REGISTER_OPERATOR(space_to_depth_grad, ops::SpaceToDepthGradOp,
                  ops::SpaceToDepthGradOpNoBuffer);
REGISTER_OP_CPU_KERNEL(
    space_to_depth,
    ops::SpaceToDepthKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SpaceToDepthKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SpaceToDepthKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    space_to_depth_grad,
    ops::SpaceToDepthGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SpaceToDepthGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SpaceToDepthGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
