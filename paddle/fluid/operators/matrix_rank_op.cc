// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/matrix_rank_op.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using DDim = framework::DDim;
DDim OutDDim(const DDim& x_dim) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  // 非batch，只是一个矩阵，还是在X前先newaxis下？？？
  if (x_vec.size() == 2) {
    return framework::make_ddim({1});
  }
  x_vec.erase(x_vec.end() - 2, x_vec.end());  // rank - 2
  return framework::make_ddim(x_vec);
}

class MatrixRankeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "MatrixRank");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MatrixRank");

    auto in_dims = ctx->GetInputDim("X");
    // 矩阵X的维度必须大于2
    PADDLE_ENFORCE_GE(in_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "the rank of input must greater than 2"));

    // 存疑，numpy、torch输入和输出广播会出现特殊情况？？？
    ctx->SetOutputDim("Out", OutDDim(in_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

// 该类可以参考PowOpMaker
class MatrixRankeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of matrix_rank op.");

    AddInput("TolTensor",
             "(Tensor<float>, optional). Tol tensor, shape is same as X batch.")
        .AsDispensable();
    AddOutput("Out", "(Tensor), The output tensor of matrix_rank op.");
    AddAttr<float>("tol", "(float, optional). tol").SetDefault(0.0f);
    AddAttr<bool>("hermitian", "(bool, optional). whether is hermitian matrix")
        .SetDefault(false);
    AddComment(R"DOC(
MatrixRank Operator.
This operator is used to perform MatrixRank operation for batched matrics.
$$out = matrix_rank(X, tol, hermitian)$$
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(matrix_rank, ops::MatrixRankeOp, ops::MatrixRankeOpMaker);

REGISTER_OP_CPU_KERNEL(matrix_rank, ops::MatrixRankCPUKernel<float>,
                       ops::MatrixRankCPUKernel<double>);