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

#include "paddle/fluid/operators/squared_l2_distance_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"

namespace paddle {
namespace operators {

class SquaredL2DistanceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SquaredL2DistanceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of SquaredL2DistanceOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("sub_result"),
        "Output(sub_result) of SquaredL2DistanceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SquaredL2DistanceOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(framework::arity(x_dims), framework::arity(y_dims),
                      "Tensor rank of both SquaredL2DistanceOp's "
                      "inputs must be same.");

    int rank = framework::arity(x_dims);
    PADDLE_ENFORCE_GE(rank, 2, "Tensor rank should be at least equal to 2.");
    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (framework::product(x_dims) <= 0 || framework::product(y_dims) <= 0)) {
      check = false;
    }
    if (check) {
      PADDLE_ENFORCE_EQ(product(x_dims) / x_dims[0],
                        product(y_dims) / y_dims[0],
                        "Product of dimensions expcet the first dimension of "
                        "input and target must be equal.");
    }
    check = true;
    if ((!ctx->IsRuntime()) && (y_dims[0] <= 0 || x_dims[0] <= 0)) {
      check = false;
    }
    if (check) {
      PADDLE_ENFORCE(y_dims[0] == 1 || y_dims[0] == x_dims[0],
                     "First dimension of target must be equal to input "
                     "or to 1.");
    }
    ctx->SetOutputDim("sub_result", {x_dims[0], product(x_dims) / x_dims[0]});
    ctx->SetOutputDim("Out", {x_dims[0], 1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(SquaredL2DistanceGradOpNoBuffer, "X",
                                      "Y");

class SquaredL2DistanceGradOpDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());

    op->SetType("squared_l2_distance_grad");

    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetInput("sub_result", Output("sub_result"));
    op->SetInput("X", Input("X"));
    op->SetInput("Y", Input("Y"));

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));

    op->SetAttrMap(Attrs());

    return op;
  }
};

class SquaredL2DistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input of SquaredL2DistanceOp.");
    AddInput("Y", "(Tensor) Target of SquaredL2DistanceOp.");
    AddOutput("sub_result",
              "(Tensor) Buffering subtraction result which "
              "will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out", "(Tensor) Squared l2 distance between input and target.");
    AddComment(R"DOC(
SquaredL2Distance operator

This operator will cacluate the squared L2 distance for the input and 
the target. Number of distance value will be equal to the first dimension 
of input. First dimension of the target could be equal to the input or to 1. 
If the first dimension of target is 1, the operator will broadcast target's 
first dimension to input's first dimension. During backward propagation, 
the user can decide whether to calculate the gradient of the input or 
the target or both.

Both the input X and Y can carry the LoD (Level of Details) information. 
However, the output only shares the LoD information with input X.
    )DOC");
  }
};

class SquaredL2DistanceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of Out should not be null");
    PADDLE_ENFORCE(ctx->HasInput("sub_result"), "SubResult should not be null");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_INFERSHAPE_ENFORCE_EQ(ctx, out_dims[0], x_dims[0],
                                 "First dimension of output gradient and "
                                 "input value must be equal.");
    PADDLE_INFERSHAPE_ENFORCE_EQ(ctx, out_dims[1], 1,
                                 "Second dimension of output gradient "
                                 "must be 1.");
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) ctx->SetOutputDim(x_grad_name, x_dims);
    if (ctx->HasOutput(y_grad_name)) ctx->SetOutputDim(y_grad_name, y_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("sub_result")->type(),
                                   ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(squared_l2_distance, ops::SquaredL2DistanceOp,
                  ops::SquaredL2DistanceOpMaker,
                  ops::SquaredL2DistanceGradOpDescMaker);
REGISTER_OPERATOR(squared_l2_distance_grad, ops::SquaredL2DistanceGradOp,
                  ops::SquaredL2DistanceGradOpNoBuffer);
REGISTER_OP_CPU_KERNEL(
    squared_l2_distance,
    ops::SquaredL2DistanceKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(squared_l2_distance_grad,
                       ops::SquaredL2DistanceGradKernel<
                           paddle::platform::CPUDeviceContext, float>);
