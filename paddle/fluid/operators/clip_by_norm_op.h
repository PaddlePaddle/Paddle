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

#pragma once

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
// using SelectedRows = phi::SelectedRows;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

class ClipByNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(X) of ClipByNormOp should not be null. Please "
                          "check if it is created correctly."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"),
                      true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of ClipByNormOp should not be null. "
                          "Please check if it is created correctly."));
    auto max_norm = ctx->Attrs().Get<float>("max_norm");
    PADDLE_ENFORCE_GT(
        max_norm,
        0,
        platform::errors::InvalidArgument("max_norm should be greater than 0. "
                                          "Received max_norm is %f.",
                                          max_norm));
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ClipByNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input of clip_by_norm op and data type is float32."
             "The number of dimensions must be between [1, 9].");
    AddOutput("Out",
              "(Tensor) The output of clip_by_norm op with shape as input(X)"
              "The data type is float32.");
    AddAttr<float>("max_norm", "(float) The maximum norm value.");
    AddComment(R"DOC(
ClipByNorm Operator.

This operator limits the L2 norm of the input $X$ within $max\_norm$.
If the L2 norm of $X$ is less than or equal to $max\_norm$, $Out$ will be
the same as $X$. If the L2 norm of $X$ is greater than $max\_norm$, $X$ will
be linearly scaled to make the L2 norm of $Out$ equal to $max\_norm$, as
shown in the following formula:

$$
Out = \\frac{max\\_norm * X}{norm(X)},
$$

where $norm(X)$ represents the L2 norm of $X$.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle
