/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/batch_norm_op.h"

namespace paddle {
namespace operators {

class BatchNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "");
    PADDLE_ENFORCE(ctx->HasInput("Scale"), "");
    PADDLE_ENFORCE(ctx->HasInput("Bias"), "");
    PADDLE_ENFORCE(ctx->HasInput("Mean"), "");
    PADDLE_ENFORCE(ctx->HasInput("Variance"), "");
    PADDLE_ENFORCE(ctx->HasOutput("MeanOut"), "");
    PADDLE_ENFORCE(ctx->HasOutput("VarianceOut"), "");
    PADDLE_ENFORCE(ctx->HasOutput("SavedMean"), "");
    PADDLE_ENFORCE(ctx->HasOutput("SavedVariance"), "");

    // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
    PADDLE_ENFORCE_EQ(ctx->Inputs("Mean"), ctx->Outputs("MeanOut"),
                      "Mean and MeanOut should share the same memory");
    PADDLE_ENFORCE_EQ(ctx->Inputs("Variance"), ctx->Outputs("VarianceOut"),
                      "Variance and VarianceOut should share the same memory");

    const auto x_dims = ctx->GetInputDim("X");
    const int C = x_dims[1];  // channel num

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], C);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], C);

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("MeanOut", {C});
    ctx->SetOutputDim("VarianceOut", {C});
    ctx->SetOutputDim("SavedMean", {C});
    ctx->SetOutputDim("SavedVariance", {C});
  }
};

class BatchNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BatchNormOpMaker(framework::OpProto* proto,
                   framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<bool>("is_test", "").SetDefault(false);
    AddAttr<float>("momentum", "").SetDefault(0.5);
    AddAttr<float>("epsilon", "").SetDefault(1e-5);
    AddInput("X", "The input 4-dimensional tensor");
    AddInput("Scale", "The second input of mul op");
    AddInput("Bias",
             "The bias as a 1-dimensional "
             "tensor of size C to be applied to the output");
    AddInput("Mean",
             "The running mean (training) or the "
             "estimated mean (testing)");
    AddInput("Variance",
             "The running variance (training) "
             "or the estimated");
    AddOutput("MeanOut",
              "The running mean (training) or the "
              "estimated mean (testing)");
    AddOutput("VarianceOut",
              "The running variance (training) "
              "or the estimated");
    AddOutput("SavedMean", "");
    AddOutput("SavedVariance", "");
    AddComment(R"DOC(
https://arxiv.org/pdf/1502.03167.pdf

NHWC `[batch, in_height, in_width, in_channels]`
NCHW `[batch, in_channels, in_height, in_width]`

we choose NCHW as the order.

)DOC");
  }
};

class BatchNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
            batch_norm_grad, ops::BatchNormGradOp);
REGISTER_OP_CPU_KERNEL(batch_norm,
                       ops::BatchNormKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradKernel<paddle::platform::CPUPlace, float>);
