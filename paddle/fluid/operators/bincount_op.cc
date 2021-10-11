/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/bincount_op.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class BincountOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of BincountOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of BincountOp should not be null."));

    auto input_dim = ctx->GetInputDim("X");
    auto minlength = ctx->Attrs().Get<int>("minlength");

    PADDLE_ENFORCE_GE(minlength, 0,
                      platform::errors::InvalidArgument(
                          "The minlength should be greater than or equal to 0."
                          "But received minlength is %d",
                          minlength));

    PADDLE_ENFORCE_EQ(input_dim.size(), 1,
                      platform::errors::InvalidArgument(
                          "The 'shape' of Input(X) must be 1-D tensor."
                          "But the dimension of Input(X) is [%d]",
                          input_dim.size()));

    if (ctx->HasInput("Weights")) {
      auto weights_dim = ctx->GetInputDim("Weights");
      PADDLE_ENFORCE_EQ(weights_dim.size(), 1,
                        platform::errors::InvalidArgument(
                            "The 'shape' of Input(Weights) must be 1-D tensor."
                            "But the dimension of Input(Weights) is [%d]",
                            weights_dim.size()));

      PADDLE_ENFORCE_EQ(
          weights_dim[0], input_dim[0],
          platform::errors::InvalidArgument(
              "The 'shape' of Input(Weights) must be equal to the 'shape' of "
              "Input(X)."
              "But received: the 'shape' of Input(Weights) is [%s],"
              "the 'shape' of Input(X) is [%s]",
              weights_dim, input_dim));
    }

    ctx->SetOutputDim("Out", framework::make_ddim({-1}));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const {
    auto data_type =
        ctx.HasInput("Weights")
            ? OperatorWithKernel::IndicateVarDataType(ctx, "Weights")
            : OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class BincountOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of Bincount op,");
    AddInput("Weights", "(Tensor) The weights tensor of Bincount op,")
        .AsDispensable();
    AddOutput("Out", "(Tensor) The output tensor of Bincount op,");
    AddAttr<int>("minlength", "(int) The minimal numbers of bins")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(R"DOC(
          Bincount Operator.
          Computes frequency of each value in the input tensor.
          Elements of input tensor should be non-negative ints.
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    bincount, ops::BincountOp, ops::BincountOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    bincount, ops::BincountKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BincountKernel<paddle::platform::CPUDeviceContext, double>,
    ops::BincountKernel<paddle::platform::CPUDeviceContext, int>,
    ops::BincountKernel<paddle::platform::CPUDeviceContext, int64_t>);
