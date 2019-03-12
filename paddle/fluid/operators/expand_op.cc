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

#include "paddle/fluid/operators/expand_op.h"
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class ExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null.");

    std::vector<int> expand_times =
        ctx->Attrs().Get<std::vector<int>>("expand_times");
    auto x_dims = ctx->GetInputDim("X");

    PADDLE_ENFORCE_EQ(static_cast<size_t>(x_dims.size()), expand_times.size(),
                      "The number of Attr(expand_times)'s value must be equal "
                      "to the rank of Input(X).");
    PADDLE_ENFORCE_LE(x_dims.size(), 6,
                      "The rank of Input(X) must not be greater than 6.");

    std::vector<int64_t> out_shape(x_dims.size());
    for (size_t i = 0; i < expand_times.size(); ++i) {
      PADDLE_ENFORCE_GE(expand_times[i], 1,
                        "Each value of Attr(expand_times) should not be "
                        "less than 1.");
      out_shape[i] = x_dims[i] * expand_times[i];
    }

    // set the first dim to -1 in compile time
    if (!ctx->IsRuntime() && x_dims[0] < 0) {
      out_shape[0] = x_dims[0];
    }

    ctx->SetOutputDim("Out", framework::make_ddim(out_shape));
    if (out_shape[0] == x_dims[0]) {
      ctx->ShareLoD("X", "Out");
    }
  }
};

class ExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After expanding, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(expand_times).");
    AddAttr<std::vector<int>>("expand_times",
                              "Expand times number for each dimension.");
    AddComment(R"DOC(
Expand operator tiles the input by given times number. You should set times
number for each dimension by providing attribute 'expand_times'. The rank of X
should be in [1, 6]. Please note that size of 'expand_times' must be the same
with X's rank. Following is a using case:

Input(X) is a 3-D tensor with shape [2, 3, 1]:

        [
           [[1], [2], [3]],
           [[4], [5], [6]]
        ]

Attr(expand_times):  [1, 2, 2]

Output(Out) is a 3-D tensor with shape [2, 6, 2]:

        [
            [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
        ]

)DOC");
  }
};

class ExpandGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> expand_times =
        ctx->Attrs().Get<std::vector<int>>("expand_times");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    size_t start_pos = 0u;
    if (!ctx->IsRuntime() && x_dims[0] < 0) {
      PADDLE_ENFORCE_EQ(
          x_dims[0], out_dims[0],
          "The first dimension size of Input(Out@GRAD) should be "
          "equal to the crroresponding dimension size of Input(X)");
      start_pos = 1u;
    }

    for (size_t i = start_pos; i < expand_times.size(); ++i) {
      PADDLE_ENFORCE_EQ(x_dims[i] * expand_times[i], out_dims[i],
                        "Each dimension size of Input(Out@GRAD) should be "
                        "equal to multiplication of crroresponding dimension "
                        "size of Input(X) and Attr(expand_times) value.");
    }

    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

class ExpandGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("expand_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(expand, ops::ExpandOp, ops::ExpandOpMaker,
                  ops::ExpandGradOpDescMaker);
REGISTER_OPERATOR(expand_grad, ops::ExpandGradOp);
REGISTER_OP_CPU_KERNEL(
    expand, ops::ExpandKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ExpandKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ExpandKernel<paddle::platform::CPUDeviceContext, bool>);
REGISTER_OP_CPU_KERNEL(
    expand_grad,
    ops::ExpandGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandGradKernel<paddle::platform::CPUDeviceContext, double>);
