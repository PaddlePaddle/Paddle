// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/matrix_rank_tol_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/funcs/values_vectors_functor.h"
#include "paddle/phi/kernels/impl/matrix_rank_kernel_impl.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T>
void LapackSVD(const T* x_data, T* eigenvalues_data, int rows, int cols) {
  char jobz = 'N';
  int mx = std::max(rows, cols);
  int mn = std::min(rows, cols);
  T* a = const_cast<T*>(x_data);  // NOLINT
  int lda = rows;
  int lwork = 3 * mn + std::max(mx, 7 * mn);
  std::vector<T> work(lwork);
  std::vector<int> iwork(8 * mn);
  int info = 0;

  phi::funcs::lapackSvd<T>(jobz,
                           rows,
                           cols,
                           a,
                           lda,
                           eigenvalues_data,
                           nullptr,
                           1,
                           nullptr,
                           1,
                           work.data(),
                           lwork,
                           iwork.data(),
                           &info);

  if (info < 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "This %s-th argument has an illegal value", info));
  }
  if (info > 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "DBDSDC/SBDSDC did not converge, updating process failed. May be you "
        "passes a invalid matrix."));
  }
}

template <typename T>
void BatchSVD(
    const T* x_data, T* eigenvalues_data, int batches, int rows, int cols) {
  int stride = rows * cols;
  int k = std::min(rows, cols);
  for (int i = 0; i < batches; ++i) {
    LapackSVD<T>(x_data + i * stride, eigenvalues_data + i * k, rows, cols);
  }
}

template <typename T, typename Context>
void MatrixRankTolKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& atol_tensor,
                         bool use_default_tol,
                         bool hermitian,
                         DenseTensor* out) {
  dev_ctx.template Alloc<int64_t>(out);
  auto dim_x = x.dims();
  auto dim_out = out->dims();
  int rows = static_cast<int>(dim_x[dim_x.size() - 2]);
  int cols = static_cast<int>(dim_x[dim_x.size() - 1]);
  PADDLE_ENFORCE_NE(
      rows,
      0,
      errors::InvalidArgument("The input Tensor x's shape[-2] should not "
                              "be 0, but received %s now.",
                              dim_x));
  PADDLE_ENFORCE_NE(
      cols,
      0,
      errors::InvalidArgument("The input Tensor x's shape[-1] should not "
                              "be 0, but received %s now.",
                              dim_x));
  if (x.numel() == 0) {
    std::vector<int64_t> out_dims_vec(dim_x.size() - 2);
    for (int i = 0; i < dim_x.size() - 2; ++i) {
      out_dims_vec[i] = dim_x[i];
    }
    out->Resize(phi::make_ddim(out_dims_vec));
    dev_ctx.template Alloc<int64_t>(out);
    return;
  }
  int k = std::min(rows, cols);
  int batches = static_cast<int>(x.numel() / (rows * cols));

  T rtol_T = 0;

  if (use_default_tol) {
    rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
  }

  DenseTensor eigenvalue_tensor;
  eigenvalue_tensor.Resize(detail::GetEigenvalueDim(dim_x, k));
  auto* eigenvalue_data = dev_ctx.template Alloc<T>(&eigenvalue_tensor);

  if (hermitian) {
    phi::funcs::MatrixEighFunctor<Context, T> functor;
    functor(dev_ctx, x, &eigenvalue_tensor, nullptr, true, false);
    phi::AbsKernel<T, Context>(dev_ctx, eigenvalue_tensor, &eigenvalue_tensor);
  } else {
    DenseTensor trans_x = phi::TransposeLast2Dim<T>(dev_ctx, x);
    auto* x_data = trans_x.data<T>();
    BatchSVD<T>(x_data, eigenvalue_data, batches, rows, cols);
  }

  DenseTensor max_eigenvalue_tensor;
  max_eigenvalue_tensor.Resize(detail::RemoveLastDim(eigenvalue_tensor.dims()));
  dev_ctx.template Alloc<T>(&max_eigenvalue_tensor);
  phi::MaxKernel<T, Context>(dev_ctx,
                             eigenvalue_tensor,
                             phi::IntArray({-1}),
                             false,
                             &max_eigenvalue_tensor);

  DenseTensor temp_rtol_tensor;
  temp_rtol_tensor =
      phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(rtol_T));

  DenseTensor rtol_tensor =
      phi::Multiply<T>(dev_ctx, temp_rtol_tensor, max_eigenvalue_tensor);

  DenseTensor tol_tensor;
  tol_tensor.Resize(dim_out);
  dev_ctx.template Alloc<T>(&tol_tensor);
  funcs::ElementwiseCompute<GreaterElementFunctor<T>, T>(
      dev_ctx,
      atol_tensor,
      rtol_tensor,
      GreaterElementFunctor<T>(),
      &tol_tensor);

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  DenseTensor compare_result;
  compare_result.Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc<int64_t>(&compare_result);
  int axis = -1;
  if (eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()) {
    funcs::ElementwiseCompute<funcs::GreaterThanFunctor<T, int64_t>, T, int>(
        dev_ctx,
        eigenvalue_tensor,
        tol_tensor,
        funcs::GreaterThanFunctor<T, int64_t>(),
        &compare_result,
        axis);
  } else {
    funcs::ElementwiseCompute<funcs::LessThanFunctor<T, int64_t>, T, int>(
        dev_ctx,
        eigenvalue_tensor,
        tol_tensor,
        funcs::LessThanFunctor<T, int64_t>(),
        &compare_result,
        axis);
  }

  phi::SumKernel<int64_t>(dev_ctx,
                          compare_result,
                          std::vector<int64_t>{-1},
                          compare_result.dtype(),
                          false,
                          out);
}

template <typename T, typename Context>
void MatrixRankAtolRtolKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& atol,
                              const paddle::optional<DenseTensor>& rtol,
                              bool hermitian,
                              DenseTensor* out) {
  auto dim_x = x.dims();
  auto dim_out = out->dims();
  int rows = static_cast<int>(dim_x[dim_x.size() - 2]);
  int cols = static_cast<int>(dim_x[dim_x.size() - 1]);
  PADDLE_ENFORCE_NE(
      rows,
      0,
      errors::InvalidArgument("The input Tensor x's shape[-2] should not "
                              "be 0, but received %s now.",
                              dim_x));
  PADDLE_ENFORCE_NE(
      cols,
      0,
      errors::InvalidArgument("The input Tensor x's shape[-1] should not "
                              "be 0, but received %s now.",
                              dim_x));
  if (x.numel() == 0) {
    std::vector<int64_t> out_dims_vec(dim_x.size() - 2);
    for (int i = 0; i < dim_x.size() - 2; ++i) {
      out_dims_vec[i] = dim_x[i];
    }
    out->Resize(phi::make_ddim(out_dims_vec));
    dev_ctx.template Alloc<int64_t>(out);
    return;
  }
  dev_ctx.template Alloc<int64_t>(out);
  int k = std::min(rows, cols);
  int batches = static_cast<int>(x.numel() / (rows * cols));

  DenseTensor eigenvalue_tensor;
  eigenvalue_tensor.Resize(detail::GetEigenvalueDim(dim_x, k));
  auto* eigenvalue_data = dev_ctx.template Alloc<T>(&eigenvalue_tensor);

  if (hermitian) {
    phi::funcs::MatrixEighFunctor<Context, T> functor;
    functor(dev_ctx, x, &eigenvalue_tensor, nullptr, true, false);
    phi::AbsKernel<T, Context>(dev_ctx, eigenvalue_tensor, &eigenvalue_tensor);
  } else {
    DenseTensor trans_x = phi::TransposeLast2Dim<T>(dev_ctx, x);
    auto* x_data = trans_x.data<T>();
    BatchSVD<T>(x_data, eigenvalue_data, batches, rows, cols);
  }

  DenseTensor max_eigenvalue_tensor;
  max_eigenvalue_tensor.Resize(detail::RemoveLastDim(eigenvalue_tensor.dims()));
  dev_ctx.template Alloc<T>(&max_eigenvalue_tensor);
  phi::MaxKernel<T, Context>(dev_ctx,
                             eigenvalue_tensor,
                             phi::IntArray({-1}),
                             false,
                             &max_eigenvalue_tensor);

  DenseTensor tol_tensor;
  tol_tensor.Resize(dim_out);
  dev_ctx.template Alloc<T>(&tol_tensor);

  if (rtol) {
    DenseTensor tmp_rtol_tensor;
    tmp_rtol_tensor = phi::Multiply<T>(dev_ctx, *rtol, max_eigenvalue_tensor);
    funcs::ElementwiseCompute<GreaterElementFunctor<T>, T>(
        dev_ctx,
        atol,
        tmp_rtol_tensor,
        GreaterElementFunctor<T>(),
        &tol_tensor);
  } else {
    // when `rtol` is specified to be None in py api
    // use rtol=eps*max(m, n) only if `atol` is passed with value 0.0, else use
    // rtol=0.0
    T rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    DenseTensor default_rtol_tensor;
    default_rtol_tensor =
        phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(rtol_T));
    default_rtol_tensor =
        phi::Multiply<T>(dev_ctx, default_rtol_tensor, max_eigenvalue_tensor);

    DenseTensor zero_tensor;
    zero_tensor = phi::FullLike<T, Context>(
        dev_ctx, default_rtol_tensor, static_cast<T>(0.0));

    DenseTensor atol_compare_result;
    atol_compare_result.Resize(default_rtol_tensor.dims());
    phi::EqualKernel<T, Context>(
        dev_ctx, atol, zero_tensor, &atol_compare_result);

    DenseTensor selected_rtol_tensor;
    selected_rtol_tensor.Resize(default_rtol_tensor.dims());
    phi::WhereKernel<T, Context>(dev_ctx,
                                 atol_compare_result,
                                 default_rtol_tensor,
                                 zero_tensor,
                                 &selected_rtol_tensor);
    funcs::ElementwiseCompute<GreaterElementFunctor<T>, T>(
        dev_ctx,
        atol,
        selected_rtol_tensor,
        GreaterElementFunctor<T>(),
        &tol_tensor);
  }

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  DenseTensor compare_result;
  compare_result.Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc<int64_t>(&compare_result);
  int axis = -1;
  if (eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()) {
    funcs::ElementwiseCompute<funcs::GreaterThanFunctor<T, int64_t>, T, int>(
        dev_ctx,
        eigenvalue_tensor,
        tol_tensor,
        funcs::GreaterThanFunctor<T, int64_t>(),
        &compare_result,
        axis);
  } else {
    funcs::ElementwiseCompute<funcs::LessThanFunctor<T, int64_t>, T, int>(
        dev_ctx,
        eigenvalue_tensor,
        tol_tensor,
        funcs::LessThanFunctor<T, int64_t>(),
        &compare_result,
        axis);
  }

  phi::SumKernel<int64_t>(dev_ctx,
                          compare_result,
                          std::vector<int64_t>{-1},
                          compare_result.dtype(),
                          false,
                          out);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    matrix_rank_tol, CPU, ALL_LAYOUT, phi::MatrixRankTolKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

PD_REGISTER_KERNEL(matrix_rank_atol_rtol,
                   CPU,
                   ALL_LAYOUT,
                   phi::MatrixRankAtolRtolKernel,
                   float,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
