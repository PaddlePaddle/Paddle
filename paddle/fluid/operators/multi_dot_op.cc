/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

/**
 * @brief compute the output shape and check the input shape valid or not
 */
inline framework::DDim ComputeAndCheckShape(
    const bool is_runtime, const std::vector<framework::DDim>& inputs_dims) {
  const size_t n = inputs_dims.size();
  auto first_dim = inputs_dims[0];

  bool is_vector = false;
  framework::DDim out_dim;

  PADDLE_ENFORCE_LT(
      first_dim.size(), static_cast<size_t>(3),
      platform::errors::InvalidArgument(
          "multi_dot: the first input tensor must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the first tensor is 1D of size n view it as a row vector (1, n)
  if (first_dim.size() == 1) {
    first_dim = phi::make_ddim({1, static_cast<int>(first_dim[0])});
    is_vector = true;
  }

  auto last_dim = inputs_dims[n - 1];
  PADDLE_ENFORCE_LT(
      last_dim.size(), static_cast<size_t>(3),
      platform::errors::InvalidArgument(
          "the last input tensor of multi_dot must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the last tensor is 1D of size n view it as a column vector (n, 1)
  if (last_dim.size() == 1) {
    last_dim = phi::make_ddim({static_cast<int>(last_dim[0]), 1});
    out_dim = is_vector ? phi::make_ddim({1}) : phi::make_ddim({first_dim[0]});
  } else {
    out_dim = is_vector ? phi::make_ddim({last_dim[1]})
                        : phi::make_ddim({first_dim[0], last_dim[1]});
  }

  auto width = first_dim[1];
  for (size_t i = 1; i < n - 1; i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(), static_cast<size_t>(2),
                      platform::errors::InvalidArgument(
                          "the input tensor of multi_dot op must be 2D."));

    const auto& tmp_dim = inputs_dims[i];
    PADDLE_ENFORCE_EQ(
        tmp_dim[0], width,
        platform::errors::InvalidArgument(
            "the input matrix does not meet the multiplication requirements."));
    width = tmp_dim[1];
  }

  PADDLE_ENFORCE_EQ(
      last_dim[0], width,
      platform::errors::InvalidArgument(
          "the input matrix does not meet the multiplication requirements."));

  return out_dim;
}

class MultiDotOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensors of multi_dot operator.").AsDuplicable();
    AddOutput("Out", "The output tensor of multi_dot operator");
    AddComment(R"DOC(
Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.

multi_dot chains MatMul and uses optimal parenthesization of the matrices [1] [2]. Depending on the shapes of the matrices, this can speed up the multiplication a lot.

If the first argument is 1-D it is treated as a row vector. If the last argument is 1-D it is treated as a column vector. The other arguments must be 2-D.
      )DOC");
  }
};

class MultiDotOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "multi_dot");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "multi_dot");

    auto inputs_dims = ctx->GetInputsDim("X");

    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_GT(
        inputs_num, static_cast<size_t>(1),
        platform::errors::InvalidArgument(
            "The number of input tensors in multi_dot op should > 1."));
    auto out_dims = ComputeAndCheckShape(ctx->IsRuntime(), inputs_dims);
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class MultiDotOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "multi_dot");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "multi_dot");

    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    auto ins_dims = ctx->GetInputsDim(in_x);
    ctx->SetOutputsDim(out_x_g_n, ins_dims);
    ctx->ShareAllLoD(in_x, out_x_g_n);
  }
};

template <typename T>
class MultiDotOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("multi_dot_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
  }
};
template <typename T>
class MultiDotOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("multi_dot");
    grad_op->SetInput("X", this->Input(("X")));
    grad_op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    grad_op->SetOutput("DDx", this->OutputGrad(framework::GradVarName("X")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(multi_dot, ops::MultiDotOp, ops::MultiDotOpMaker,
                  ops::MultiDotOpGradMaker<paddle::framework::OpDesc>,
                  ops::MultiDotOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(multi_dot_grad, ops::MultiDotOpGrad,
                  ops::MultiDotOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MultiDotOpDoubleGradMaker<paddle::imperative::OpBase>);
