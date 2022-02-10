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

#include "paddle/fluid/operators/matrix_rank_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/svd_helper.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {
using DDim = framework::DDim;

namespace detail {
static DDim CheckAndGetOutputDim(const DDim& dim_x) {
  auto x_vec = framework::vectorize(dim_x);
  if (x_vec.size() == 2) {
    return framework::make_ddim({1});
  }
  x_vec.erase(x_vec.end() - 2, x_vec.end());
  return framework::make_ddim(x_vec);
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
        GetBroadcastDimsArrays(dim_x_batch, dim_tol, x_batch_dims_array.data(),
                               tol_dims_array.data(), out_dims_array.data(),
                               max_dim, axis);
        ctx->SetOutputDim("Out", framework::make_ddim(out_dims_array));
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

template <typename T>
void BatchEigenvalues(const T* x_data, T* eigenvalues_data, int batches,
                      int rows, int cols, int k) {
  // Eigen::Matrix API need non-const pointer.
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, rows);
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m);
    auto eigenvalues = eigen_solver.eigenvalues().cwiseAbs();
    for (int j = 0; j < k; j++) {
      *(eigenvalues_data + i * k + j) = eigenvalues[j];
    }
  }
}

template <typename T>
void BatchSVD(const T* x_data, T* eigenvalues_data, int batches, int rows,
              int cols, int k) {
  // Eigen::Matrix API need non-const pointer.
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  Eigen::BDCSVD<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      svd;
  for (int i = 0; i < batches; i++) {
    auto m = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, cols);
    svd.compute(m);
    auto res_s = svd.singularValues();
    for (int j = 0; j < k; j++) {
      eigenvalues_data[i * k + j] = res_s[j];
    }
  }
}

template <typename T>
class MatrixRankCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    auto* x_data = x->data<T>();
    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<int64_t>(context.GetPlace());
    bool hermitian = context.Attr<bool>("hermitian");

    auto dim_x = x->dims();
    auto dim_out = out->dims();
    int rows = dim_x[dim_x.size() - 2];
    int cols = dim_x[dim_x.size() - 1];
    int k = std::min(rows, cols);
    auto numel = x->numel();
    int batches = numel / (rows * cols);

    bool use_default_tol = context.Attr<bool>("use_default_tol");
    const Tensor* atol_tensor = nullptr;
    Tensor temp_tensor;
    T rtol_T = 0;
    if (use_default_tol) {
      framework::TensorFromVector<T>(std::vector<T>{0},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
      rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    } else if (context.HasInput("TolTensor")) {
      atol_tensor = context.Input<Tensor>("TolTensor");
    } else {
      framework::TensorFromVector<T>(std::vector<T>{context.Attr<float>("tol")},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
    }

    Tensor eigenvalue_tensor;
    auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
        detail::GetEigenvalueDim(dim_x, k), context.GetPlace());
    if (hermitian) {
      BatchEigenvalues<T>(x_data, eigenvalue_data, batches, rows, cols, k);
    } else {
      BatchSVD<T>(x_data, eigenvalue_data, batches, rows, cols, k);
    }

    auto dito_T =
        math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext, T>(
            context);
    std::vector<int> max_eigenvalue_shape = framework::vectorize<int>(
        detail::RemoveLastDim(eigenvalue_tensor.dims()));
    Tensor max_eigenvalue_tensor =
        dito_T.ReduceMax(eigenvalue_tensor, max_eigenvalue_shape);

    Tensor temp_rtol_tensor;
    framework::TensorFromVector<T>(std::vector<T>{rtol_T}, &temp_rtol_tensor);
    Tensor rtol_tensor = dito_T.Mul(temp_rtol_tensor, max_eigenvalue_tensor);
    Tensor tol_tensor;
    tol_tensor.mutable_data<T>(dim_out, context.GetPlace());
    ElementwiseComputeEx<GreaterElementFunctor<T>, platform::CPUDeviceContext,
                         T, T>(context, atol_tensor, &rtol_tensor, -1,
                               GreaterElementFunctor<T>(), &tol_tensor);

    tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

    Tensor compare_result;
    compare_result.mutable_data<int64_t>(detail::NewAxisDim(dim_out, k),
                                         context.GetPlace());

    int axis = -1;
    if (eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()) {
      ElementwiseComputeEx<GreaterThanFunctor<T, int64_t>,
                           platform::CPUDeviceContext, T, int>(
          context, &eigenvalue_tensor, &tol_tensor, axis,
          GreaterThanFunctor<T, int64_t>(), &compare_result);
    } else {
      ElementwiseComputeEx<LessThanFunctor<T, int64_t>,
                           platform::CPUDeviceContext, T, int>(
          context, &eigenvalue_tensor, &tol_tensor, axis,
          LessThanFunctor<T, int64_t>(), &compare_result);
    }
    auto dito_int =
        math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext,
                                                 int64_t>(context);
    std::vector<int> result_shape = framework::vectorize<int>(dim_out);
    Tensor result = dito_int.ReduceSum(compare_result, result_shape);
    out->ShareDataWith(result);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(matrix_rank, ops::MatrixRankeOp, ops::MatrixRankeOpMaker);

REGISTER_OP_CPU_KERNEL(matrix_rank, ops::MatrixRankCPUKernel<float>,
                       ops::MatrixRankCPUKernel<double>);
