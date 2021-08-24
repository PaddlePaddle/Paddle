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

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/elementwise/svd_helper.h"
#include "paddle/fluid/operators/matrix_rank_op.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {
using DDim = framework::DDim;

DDim InputBatchDim(const DDim& dim_x) {
  auto x_vec = framework::vectorize(dim_x);
  // 非batch，单个二维矩阵
  if (x_vec.size() == 2) {
    return framework::make_ddim({1});
  }
  x_vec.erase(x_vec.end() - 2, x_vec.end());  // rank - 2
  return framework::make_ddim(x_vec);
}

DDim EigenvalueDim(const DDim& dim, int k) {
  auto vec = framework::vectorize(dim);
  vec.erase(vec.end() - 2, vec.end());
  vec.push_back(k);
  return framework::make_ddim(vec);
}

DDim NewAxisDim(const DDim& dim, int k) {
  auto vec = framework::vectorize(dim);
  vec.push_back(k);
  return framework::make_ddim(vec);
}

DDim RemoveLastAndNewAxisDim(const DDim& dim, int k) {
  auto vec = framework::vectorize(dim);
  vec.erase(vec.end() - 1, vec.end());
  vec.push_back(k);
  return framework::make_ddim(vec);
}

DDim RemoveLastDim(const DDim& dim) {
  auto vec = framework::vectorize(dim);
  if (vec.size() == 1) {
    return framework::make_ddim({1});
  }
  vec.erase(vec.end() - 1, vec.end());
  return framework::make_ddim(vec);
}

DDim UDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = framework::vectorize(x_dim);
  x_vec[x_vec.size() - 1] = k;
  return framework::make_ddim(x_vec);
}

DDim VHDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = framework::vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  return framework::make_ddim(x_vec);
}

// framework::OpKernelType KernelType(const framework::ExecutionContext& ctx,
//                                       const framework::OperatorWithKernel&
//                                       oper,
//                                       const std::string& name) {
//   framework::LibraryType library{framework::LibraryType::kPlain};
//   framework::DataLayout layout = framework::DataLayout::kAnyLayout;
//   auto data_type = oper.IndicateVarDataType(ctx, name);
// #ifdef PADDLE_WITH_MKLDNN
//   auto it = oper.Attrs().find("use_mkldnn");
//   if (library == framework::LibraryType::kPlain && it != oper.Attrs().end()
//   &&
//       oper.CanMKLDNNBeUsed(ctx, data_type)) {
//     library = framework::LibraryType::kMKLDNN;
//     layout = framework::DataLayout::kMKLDNN;
//   }
// #endif
//   return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
// }

class MatrixRankeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "MatrixRank");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MatrixRank");
    auto dim_x = ctx->GetInputDim("X");
    // 矩阵X的维度必须大于2
    PADDLE_ENFORCE_GE(dim_x.size(), 2,
                      platform::errors::InvalidArgument(
                          "the rank of input must greater than 2"));
    // if hermitian == true, matrix should be n*n
    bool hermitian = ctx->Attrs().Get<bool>("hermitian");
    if (hermitian) {
      int rows = dim_x[dim_x.size() - 2];
      int cols = dim_x[dim_x.size() - 1];
      PADDLE_ENFORCE_EQ(rows, cols,
                        platform::errors::InvalidArgument(
                            "if hermitian == true, rows == cols for matrix"));
    }

    DDim dim_x_batch = InputBatchDim(dim_x);
    if (ctx->Attrs().Get<bool>(
            "use_default_tol")) {  // user not input TolTensor and tol
      ctx->SetOutputDim("Out", dim_x_batch);
    } else if (ctx->HasInput("TolTensor")) {
      auto dim_tol = ctx->GetInputDim("TolTensor");
      // VLOG(3) << "dim_x_batch: " << dim_x_batch;
      // VLOG(3) << "dim_tol: " << dim_tol;

      if (dim_x_batch == dim_tol) {
        // VLOG(3) << "dim_x_batch == dim_tol";
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
        for (auto& it : out_dims_array) {
          VLOG(3) << "out dims: " << it;
        }
        ctx->SetOutputDim("Out", framework::make_ddim(out_dims_array));
      }
    } else {
      ctx->SetOutputDim("Out", dim_x_batch);
    }

    ctx->ShareLoD("X", /*->*/ "Out");

    // if (ctx->HasInput("TolTensor")) {
    //   auto dim_tol = ctx->GetInputDim("TolTensor");
    //   // VLOG(3) << "dim_x_batch: " << dim_x_batch;
    //   // VLOG(3) << "dim_tol: " << dim_tol;

    //   if (dim_x_batch == dim_tol) {
    //     // VLOG(3) << "dim_x_batch == dim_tol";
    //     ctx->SetOutputDim("Out", dim_x_batch);
    //   } else {
    //     int max_dim = std::max(dim_x_batch.size(), dim_tol.size());
    //     int axis = std::abs(dim_x_batch.size() - dim_tol.size());
    //     std::vector<int> x_batch_dims_array(max_dim);
    //     std::vector<int> tol_dims_array(max_dim);
    //     std::vector<int> out_dims_array(max_dim);
    //     GetBroadcastDimsArrays(dim_x_batch, dim_tol,
    //     x_batch_dims_array.data(),
    //                     tol_dims_array.data(), out_dims_array.data(),
    //                     max_dim, axis);
    //     for (auto &it: out_dims_array) {
    //       VLOG(3) << "out dims: " << it;
    //     }
    //     ctx->SetOutputDim("Out", framework::make_ddim(out_dims_array));
    //   }
    // } else {
    //   ctx->SetOutputDim("Out", dim_x_batch);
    // }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
    // return KernelType(ctx, *this, "X");
  }

  // framework::OpKernelType GetKernelTypeForVar(
  //     const std::string& var_name, const Tensor& tensor,
  //     const framework::OpKernelType& expected_kernel_type) const override {
  //   if (var_name == "TolTensor") {
  //     return expected_kernel_type;
  //   }
  //   return framework::OpKernelType(expected_kernel_type.data_type_,
  //                                  tensor.place(), tensor.layout());
  // }
};

// 该类可以参考PowOpMaker
class MatrixRankeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of matrix_rank op.");
    AddInput("TolTensor", "(optional) Tol tensor, shape is same as X batch.")
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
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    // compute eigenvalues
    // VLOG(3) << "compute eigenvalues";
    auto m = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, rows);
    // VLOG(3) << m;

    // m.eigenvalues() == torch.linalg.eigvals()
    // m.selfadjointView<Eigen::Lower>().eigenvalues() == m.eigenvalues() ==
    // torch.linalg.eigvalsh()
    // eigvalsh() is used in torch.linalg.matrix_rank()
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m);
    auto eigenvalues = eigen_solver.eigenvalues().cwiseAbs();
    // 为什么这样调用不可以？？
    // auto eigenvalues =
    // m.selfadjointView<Eigen::Lower>().eigenvalues().cwiseAbs();
    // VLOG(3) << "auto eigenvalues: " << eigenvalues;
    for (int j = 0; j < k; j++) {
      // 不能用下标的方式访问吗？？
      *(eigenvalues_data + i * k + j) = eigenvalues[j];
      // eigenvalues_data[i*k+j] = eigenvalues[j];
      // VLOG(3) << "eigenvalues_data[i*k+j]: " << *(eigenvalues_data+i*k+j);
    }
  }
}

template <typename T>
void BatchSVD(const T* x_data, T* eigenvalues_data, int batches, int rows,
              int cols, int k) {
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  Eigen::BDCSVD<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      svd;
  for (int i = 0; i < batches; i++) {
    // compute SVD
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

// template <typename T>
// struct CompareFunctor {
//   HOSTDEVICE int operator()(const T& a, const T& b) const { return a > b; }
// };

// template <typename T>
// struct InverseCompareFunctor {
//   HOSTDEVICE int operator()(const T& a, const T& b) const { return a < b; }
// };

// template <typename T>
// struct GreaterElementFunctor {
//   HOSTDEVICE float operator()(const T& a, const T& b) const {
//     if (a > b) {
//       return a;
//     } else {
//       return b;
//     }
//   }
// };

template <typename T>
class MatrixRankCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get input/output
    const Tensor* x = context.Input<Tensor>("X");
    auto* x_data = x->data<T>();
    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<int64_t>(context.GetPlace());
    bool hermitian = context.Attr<bool>("hermitian");
    // get shape
    auto dim_x = x->dims();
    auto dim_out = out->dims();
    int rows = dim_x[dim_x.size() - 2];
    int cols = dim_x[dim_x.size() - 1];
    int k = std::min(rows, cols);
    auto numel = x->numel();
    int batches = numel / (rows * cols);
    // get tol
    bool use_default_tol = context.Attr<bool>("use_default_tol");
    VLOG(3) << "use_default_tol" << use_default_tol;
    const Tensor* atol_tensor = nullptr;
    Tensor temp_tensor;
    T rtol_T = 0;
    if (use_default_tol) {
      VLOG(3) << "not has tol";
      framework::TensorFromVector<T>(std::vector<T>{0},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
      rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    } else if (context.HasInput("TolTensor")) {
      VLOG(3) << "has toltensor";
      atol_tensor = context.Input<Tensor>("TolTensor");
    } else {
      VLOG(3) << "has tol";
      VLOG(3) << context.Attr<float>("tol");
      framework::TensorFromVector<T>(std::vector<T>{context.Attr<float>("tol")},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
    }

    // if (context.HasInput("TolTensor")) {
    //   VLOG(3) << "has toltensor";
    //   atol_tensor = context.Input<Tensor>("TolTensor");
    // } else if (context.HasAttr("tol")) {
    //   VLOG(3) << "has tol";
    //   // Tensor temp_tensor;
    //   VLOG(3) << context.Attr<float>("tol");
    //   framework::TensorFromVector<T>(std::vector<T>{context.Attr<float>("tol")},
    //   &temp_tensor);
    //   atol_tensor = &temp_tensor;
    // } else {
    //   VLOG(3) << "not has tol";
    //   // Tensor temp_tensor;
    //   framework::TensorFromVector<T>(std::vector<T>{0}, &temp_tensor);
    //   atol_tensor = &temp_tensor;
    //   rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    // }

    // VLOG(3) << "atol_tensor shape: " << atol_tensor->dims();
    // for (int i = 0; i < atol_tensor->numel(); i++) {
    //   VLOG(3) << "atol_tensor: " << atol_tensor->data<T>()[i];
    // }

    // compute eigenvalue/svd
    Tensor eigenvalue_tensor;
    auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
        EigenvalueDim(dim_x, k), context.GetPlace());
    if (hermitian) {
      VLOG(3) << "hermitian == true";
      BatchEigenvalues<T>(x_data, eigenvalue_data, batches, rows, cols, k);
    } else {
      BatchSVD<T>(x_data, eigenvalue_data, batches, rows, cols, k);
    }

    VLOG(3) << "eigenvalue_tensor dim: " << eigenvalue_tensor.dims()
            << std::endl;
    for (int i = 0; i < eigenvalue_tensor.numel(); i++) {
      VLOG(3) << "eigenvalue_tensor: " << eigenvalue_data[i] << std::endl;
    }

    // compare atol(absolute tol) with rtol(relative tol)
    // T rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    // // user enter tol param, use TolTensor as tol
    // if (hasTol) {
    //   rtol_T = 0;
    // }
    auto dito_T =
        math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext, T>(
            context);
    std::vector<int> max_eigenvalue_shape =
        framework::vectorize<int>(RemoveLastDim(eigenvalue_tensor.dims()));
    Tensor max_eigenvalue_tensor =
        dito_T.reduce_max(eigenvalue_tensor, max_eigenvalue_shape);

    VLOG(3) << "max_eigenvalue_tensor shape: " << max_eigenvalue_tensor.dims();
    for (int i = 0; i < max_eigenvalue_tensor.numel(); i++) {
      VLOG(3) << "max_eigenvalue_tensor: "
              << max_eigenvalue_tensor.data<T>()[i];
    }

    Tensor temp_rtol_tensor;
    framework::TensorFromVector<T>(std::vector<T>{rtol_T}, &temp_rtol_tensor);
    // rtol_tensor.mutable_data<T>(max_eigenvalue_tensor.dims(),
    // context.GetPlace());
    Tensor rtol_tensor = dito_T.mul(temp_rtol_tensor, max_eigenvalue_tensor);

    VLOG(3) << "rtol_tensor shape: " << rtol_tensor.dims();
    for (int i = 0; i < rtol_tensor.numel(); i++) {
      VLOG(3) << "rtol_tensor: " << rtol_tensor.data<T>()[i];
    }

    Tensor tol_tensor;
    tol_tensor.mutable_data<T>(dim_out, context.GetPlace());
    ElementwiseComputeEx<GreaterElementFunctor<T>, platform::CPUDeviceContext,
                         T, T>(context, atol_tensor, &rtol_tensor, -1,
                               GreaterElementFunctor<T>(), &tol_tensor);

    // if (atol_tensor->dims().size() >= rtol_tensor.dims().size()) {
    //   VLOG(3) << "atol_tensor->dims().size() >= rtol_tensor.dims().size()";
    //   ElementwiseComputeEx<CompareRetEleFunctor<T>,
    //   platform::CPUDeviceContext, T, T>(
    //                                                   context, atol_tensor,
    //                                                   &rtol_tensor, -1,
    //                                                   CompareRetEleFunctor<T>(),
    //                                                   &tol_tensor);
    // } else {
    //   VLOG(3) << "atol_tensor->dims().size() < rtol_tensor.dims().size()";
    //   ElementwiseComputeEx<InverseCompareRetEleFunctor<T>,
    //   platform::CPUDeviceContext, T, T>(
    //                                                   context, atol_tensor,
    //                                                   &rtol_tensor, -1,
    //                                                   InverseCompareRetEleFunctor<T>(),
    //                                                   &tol_tensor);
    // }

    VLOG(3) << "tol_tensor shape: " << tol_tensor.dims();
    for (int i = 0; i < tol_tensor.numel(); i++) {
      VLOG(3) << "tol_tensor: " << tol_tensor.data<T>()[i];
    }

    // add new axis last dim
    tol_tensor.Resize(NewAxisDim(tol_tensor.dims(), 1));
    // Tensor temp_atol_tensor;
    // temp_atol_tensor.ShareDataWith(*atol_tensor);
    // auto shape = framework::vectorize(dim_atol_tensor);
    // shape.insert(shape.end(), 1);
    // temp_atol_tensor.Resize(framework::make_ddim(shape));
    // compare result
    Tensor compare_result;
    compare_result.mutable_data<int>(NewAxisDim(dim_out, k),
                                     context.GetPlace());
    int axis = -1;

    // VLOG(3) << "atol_tensor dim: " << atol_tensor->dims();
    // VLOG(3) << "temp_atol_tensor dim: " << temp_atol_tensor.dims();
    if (eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()) {
      VLOG(3) << "eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()";
      // 为什么找不到comapre_op.h中的GreaterThanFunctor？
      ElementwiseComputeEx<CompareFunctor<T>, platform::CPUDeviceContext, T,
                           int>(context, &eigenvalue_tensor, &tol_tensor, axis,
                                CompareFunctor<T>(), &compare_result);
      // VLOG(3) << "compare_result dim: " << compare_result.dims() <<
      // std::endl;
      // for (int i = 0; i < compare_result.numel(); i++) {
      //     VLOG(3) << "compare_result: " << compare_result.data<int64_t>()[i]
      //     << std::endl;
      // }
      // auto dito_int =
      // math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext,
      // int64_t>(context);
      // std::vector<int> res_shape = framework::vectorize(dim_out);
      // Tensor res = dito_int.reduce_sum(compare_result, res_shape);
      // VLOG(3) << "res dim: " << res.dims() << std::endl;
      // for (int i = 0; i < res.numel(); i++) {
      //     VLOG(3) << "res: " << res.data<int64_t>()[i] << std::endl;
      // }
    } else {
      VLOG(3) << "eigenvalue_tensor.dims().size() < tol_tensor.dims().size()";
      ElementwiseComputeEx<InverseCompareFunctor<T>, platform::CPUDeviceContext,
                           T, int>(context, &eigenvalue_tensor, &tol_tensor,
                                   axis, InverseCompareFunctor<T>(),
                                   &compare_result);
    }
    // VLOG(3) << "compare_result dim: " << compare_result.dims() << std::endl;
    // for (int i = 0; i < compare_result.numel(); i++) {
    //     VLOG(3) << "compare_result: " << compare_result.data<int64_t>()[i] <<
    //     std::endl;
    // }
    auto dito_int =
        math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext,
                                                 int64_t>(context);
    std::vector<int> res_shape = framework::vectorize<int>(dim_out);
    Tensor res = dito_int.reduce_sum(compare_result, res_shape);
    out->ShareDataWith(res);
    // VLOG(3) << "res dim: " << res.dims() << std::endl;
    // for (int i = 0; i < res.numel(); i++) {
    //   VLOG(3) << "res: " << res.data<int64_t>()[i] << std::endl;
    // }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(matrix_rank, ops::MatrixRankeOp, ops::MatrixRankeOpMaker);

REGISTER_OP_CPU_KERNEL(matrix_rank, ops::MatrixRankCPUKernel<float>,
                       ops::MatrixRankCPUKernel<double>);
