/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/layer_norm_op.h"

namespace paddle {
namespace operators {

class LayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scale"),
                   "Input(Scale) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Bias"),
                   "Input(Bias) of LayerNormOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LayerNormOp should not be null.");

    auto input_dim = ctx->GetInputDim("X");
    auto scale_dim = ctx->GetInputDim("Scale");
    auto bias_dim = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(scale_dim, bias_dim,
                      "Scale and Bias input of LayerNormOp should have "
                      "should have same dimension");
    // TODO: Add more asserts
    ctx->SetOutputDim("Out", input_dim);
  }
};

class LayerNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LayerNormOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor), the input(X) is a Tensor of shape (B x D), "
             "where B is the size of the current mini-batch and D is the "
             "input data dimension.");
    AddInput("Scale",
             "(Tensor), the input(Scale) is a Tensor of "
             "shape (D x 1).");
    AddInput("Bias",
             "(Tensor), the input(Bias) is a Tensor of "
             "shape (D x 1).");

    AddOutput("Out",
              "(Tensor), the output(Out) is a Tensor of shape (B x D), "
              ", which is same as the shape of the input tensor, "
              "where B is the size of the current mini-batch and D is the "
              "input data dimension.");

    AddAttr<float>("epsilon",
                   "(float, default 1e-5) Constant for "
                   "numerical stability")
        .SetDefault(1.0e-5);
    AddComment(R"DOC(
LayerNorm Operator.

TODO: Add more details...

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(layer_norm, ops::LayerNormOp,
                             ops::LayerNormOpMaker);
REGISTER_OP_CPU_KERNEL(
    layer_norm,
    ops::LayerNormOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LayerNormOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LayerNormGradKernel<paddle::platform::CPUDeviceContext, double>);
