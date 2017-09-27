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

#include "paddle/operators/sum_op.h"
#include <vector>

namespace paddle {
namespace operators {
using framework::Tensor;

class SumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    auto x_dims = ctx->GetInputsDim("X");
    PADDLE_ENFORCE(!x_dims.empty(), "Input(X) of SumOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SumOp should not be null.");

    auto in_dim = x_dims[0];
    size_t N = x_dims.size();
    PADDLE_ENFORCE_GT(N, 1, "Input tensors count should > 1.");
    for (size_t i = 1; i < N; i++) {
      auto dim = x_dims[i];
      PADDLE_ENFORCE(in_dim == dim, "Input tensors must have same shape");
    }
    ctx->SetOutputDim("Out", in_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class SumOpMaker : public framework::OpInfoMaker {
 public:
  SumOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpInfoMaker(proto, op_checker) {
    AddInput("X", "the input tensors of sum operator.").AsDuplicable();
    AddOutput("Out", "the output tensor of sum operator.");
    AddComment(R"DOC(
Sum the input tensors.

All the inputs can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with the first input.
)DOC");
  }
};

class SumGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    auto out_grad_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_grad_names = ctx->Outputs(framework::GradVarName("X"));
    size_t x_length = x_grad_names.size();
    std::vector<framework::DDim> x_grad_dims;
    x_grad_dims.reserve(x_length);
    for (size_t i = 0; i < x_length; ++i) {
      x_grad_dims.push_back(out_grad_dims);
    }
    ctx->SetOutputsDim(framework::GradVarName("X"), x_grad_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sum, ops::SumOp, ops::SumOpMaker, sum_grad, ops::SumGradOp);
REGISTER_OP_CPU_KERNEL(sum, ops::SumKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(sum_grad,
                       ops::SumGradKernel<paddle::platform::CPUPlace, float>);
