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

#include "paddle/fluid/operators/mul_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class MulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of MulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) of MulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MulOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
    int y_num_col_dims = ctx->Attrs().Get<int>("y_num_col_dims");

    VLOG(3) << "mul operator x.shape=" << x_dims << " y.shape=" << y_dims
            << " x_num_col_dims=" << x_num_col_dims
            << " y_num_col_dims=" << y_num_col_dims;

    PADDLE_ENFORCE_GT(
        x_dims.size(), x_num_col_dims,
        "The input tensor X's rank of MulOp should be larger than "
        "x_num_col_dims.");
    PADDLE_ENFORCE_GT(
        y_dims.size(), y_num_col_dims,
        "The input tensor Y's rank of MulOp should be larger than "
        "y_num_col_dims: %ld vs %ld",
        y_dims.size(), y_num_col_dims);

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
    auto y_mat_dims = framework::flatten_to_2d(y_dims, y_num_col_dims);

    PADDLE_ENFORCE_EQ(x_mat_dims[1], y_mat_dims[0],
                      "First matrix's width must be equal with second matrix's "
                      "height. %s, %s",
                      x_mat_dims[1], y_mat_dims[0]);
    std::vector<int64_t> output_dims;
    output_dims.reserve(
        static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

    for (int i = 0; i < x_num_col_dims; ++i) {
      output_dims.push_back(x_dims[i]);
    }

    for (int i = y_num_col_dims; i < y_dims.size(); ++i) {
      output_dims.push_back(y_dims[i]);
    }

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of mul op.");
    AddInput("Y", "(Tensor), The second input tensor of mul op.");
    AddOutput("Out", "(Tensor), The output tensor of mul op.");
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two
              dimensions as its inputs. If the input $X$ is a tensor with more
              than two dimensions, $X$ will be flattened into a two-dimensional
              matrix first. The flattening rule is: the first `num_col_dims`
              will be flattened to form the first dimension of the final matrix
              (the height of the matrix), and the rest `rank(X) - num_col_dims`
              dimensions are flattened to form the second dimension of the final
              matrix (the width of the matrix). As a result, height of the
              flattened matrix is equal to the product of $X$'s first
              `x_num_col_dims` dimensions' sizes, and width of the flattened
              matrix is equal to the product of $X$'s last `rank(x) - num_col_dims`
              dimensions' size. For example, suppose $X$ is a 6-dimensional
              tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
              Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
              [24, 30].
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two,
              dimensions as its inputs. If the input $Y$ is a tensor with more
              than two dimensions, $Y$ will be flattened into a two-dimensional
              matrix first. The attribute `y_num_col_dims` determines how $Y$ is
              flattened. See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddComment(R"DOC(
Mul Operator.

This operator is used to perform matrix multiplication for input $X$ and $Y$.

The equation is:

$$Out = X * Y$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};

class MulOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};

class MulGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

class MulOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> retv(new framework::OpDesc());
    retv->SetType("mul_grad");
    retv->SetInput("X", Input("X"));
    retv->SetInput("Y", Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    retv->SetAttrMap(Attrs());
    return retv;
  }
};

class MulDoubleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("DOut"), "Input(DOut) should not be null");

    if (ctx->HasOutput("DDOut") && ctx->HasInput("DDX")) {
      ctx->ShareDim("DOut", "DDOut");
    }
    if (ctx->HasOutput("DX") && ctx->HasInput("DDY")) {
      ctx->ShareDim("X", "DX");
    }
    if (ctx->HasOutput("DY") && ctx->HasInput("DDX")) {
      ctx->ShareDim("Y", "DY");
    }
  }
};

class MulDoubleGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> retv(new framework::OpDesc());
    retv->SetType("mul_grad_grad");

    retv->SetInput("X", Input("X"));
    retv->SetInput("Y", Input("Y"));
    retv->SetInput("DOut", Input(framework::GradVarName("Out")));
    retv->SetInput("DDX", OutputGrad(framework::GradVarName("X")));
    retv->SetInput("DDY", OutputGrad(framework::GradVarName("Y")));

    auto ddx = OutputGrad(framework::GradVarName("X"));
    auto ddw = OutputGrad(framework::GradVarName("Y"));
    std::vector<std::string> empty_str = {};

    retv->SetOutput("DDOut", (ddx.empty())
                                 ? empty_str
                                 : InputGrad(framework::GradVarName("Out")));
    retv->SetOutput("DX", ddw.empty() ? empty_str : InputGrad("X"));
    retv->SetOutput("DY", ddx.empty() ? empty_str : InputGrad("Y"));

    retv->SetAttrMap(Attrs());
    return retv;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mul, ops::MulOp, ops::MulOpMaker, ops::MulOpInferVarType,
                  ops::MulOpGradMaker);
REGISTER_OPERATOR(mul_grad, ops::MulGradOp, ops::MulDoubleGradMaker);
REGISTER_OPERATOR(mul_grad_grad, ops::MulDoubleGradOp);
REGISTER_OP_CPU_KERNEL(
    mul, ops::MulKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MulKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    mul_grad, ops::MulGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MulGradKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    mul_grad_grad,
    ops::MulDoubleGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MulDoubleGradKernel<paddle::platform::CPUDeviceContext, double>);
