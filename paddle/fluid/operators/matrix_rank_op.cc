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
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {
using DDim = framework::DDim;

namespace detail {
static DDim CheckAndGetOutputDim(const DDim& dim_x) {
  auto x_vec = phi::vectorize(dim_x);
  if (x_vec.size() == 2) {
    return phi::make_ddim({1});
  }
  x_vec.erase(x_vec.end() - 2, x_vec.end());
  return phi::make_ddim(x_vec);
}
}  // namespace detail

class MatrixRankeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "MatrixRank");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MatrixRank");
    auto dim_x = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(dim_x.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dims of input must be greater than 2"));

    bool hermitian = ctx->Attrs().Get<bool>("hermitian");
    if (hermitian) {
      int rows = dim_x[dim_x.size() - 2];
      int cols = dim_x[dim_x.size() - 1];
      PADDLE_ENFORCE_EQ(rows, cols,
                        platform::errors::InvalidArgument(
                            "if hermitian == true, matrix should be n*n"));
    }

    DDim dim_x_batch = detail::CheckAndGetOutputDim(dim_x);
    if (ctx->HasInput("TolTensor")) {
      auto dim_tol = ctx->GetInputDim("TolTensor");
      if (dim_x_batch == dim_tol) {
        ctx->SetOutputDim("Out", dim_x_batch);
      } else {
        int max_dim = std::max(dim_x_batch.size(), dim_tol.size());
        int axis = std::abs(dim_x_batch.size() - dim_tol.size());
        std::vector<int> x_batch_dims_array(max_dim);
        std::vector<int> tol_dims_array(max_dim);
        std::vector<int> out_dims_array(max_dim);
        phi::funcs::GetBroadcastDimsArrays(
            dim_x_batch, dim_tol, x_batch_dims_array.data(),
            tol_dims_array.data(), out_dims_array.data(), max_dim, axis);
        ctx->SetOutputDim("Out", phi::make_ddim(out_dims_array));
      }
    } else {
      ctx->SetOutputDim("Out", dim_x_batch);
    }
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

class MatrixRankeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of matrix_rank op.");
    AddInput("TolTensor",
             "(optional) Tol tensor, shape is same as X batch or can broadcast "
             "with X batch.")
        .AsDispensable();
    AddOutput("Out", "(Tensor), The output tensor of matrix_rank op.");
    AddAttr<float>("tol", "(float, optional). tol").SetDefault(0.0f);
    AddAttr<bool>("use_default_tol",
                  "represent whether user input TolTensor/tol, if input "
                  "TolTensor/tol use_default_tol=true, otherwise "
                  "use_default_tol=false")
        .SetDefault(true);
    AddAttr<bool>("hermitian", "(bool, optional). whether is hermitian matrix")
        .SetDefault(false);
    AddComment(R"DOC(MatrixRank Operator.
    This operator is used to perform MatrixRank operation for batched matrics.
    $$out = matrix_rank(X, tol, hermitian)$$
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(matrix_rank, ops::MatrixRankeOp, ops::MatrixRankeOpMaker);
