// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/rocsolver.h"
#else
#include "paddle/phi/backends/dynload/cusolver.h"
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/lu_kernel_impl.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"

namespace phi {

#ifdef PADDLE_WITH_HIP
template <typename T>
void rocsolver_getrs(const solverHandle_t& handle,
                     rocblas_operation trans,
                     int n,
                     int nrhs,
                     T* a,
                     int lda,
                     int* ipiv,
                     T* b,
                     int ldb);

template <>
void rocsolver_getrs<float>(const solverHandle_t& handle,
                            rocblas_operation trans,
                            int n,
                            int nrhs,
                            float* a,
                            int lda,
                            int* ipiv,
                            float* b,
                            int ldb) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::rocsolver_sgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb));
}

template <>
void rocsolver_getrs<double>(const solverHandle_t& handle,
                             rocblas_operation trans,
                             int n,
                             int nrhs,
                             double* a,
                             int lda,
                             int* ipiv,
                             double* b,
                             int ldb) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::rocsolver_dgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb));
}

template <>
void rocsolver_getrs<dtype::complex<float>>(const solverHandle_t& handle,
                                            rocblas_operation trans,
                                            int n,
                                            int nrhs,
                                            dtype::complex<float>* a,
                                            int lda,
                                            int* ipiv,
                                            dtype::complex<float>* b,
                                            int ldb) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::rocsolver_cgetrs(handle,
                                trans,
                                n,
                                nrhs,
                                reinterpret_cast<rocblas_float_complex*>(a),
                                lda,
                                ipiv,
                                reinterpret_cast<rocblas_float_complex*>(b),
                                ldb));
}

template <>
void rocsolver_getrs<dtype::complex<double>>(const solverHandle_t& handle,
                                             rocblas_operation trans,
                                             int n,
                                             int nrhs,
                                             dtype::complex<double>* a,
                                             int lda,
                                             int* ipiv,
                                             dtype::complex<double>* b,
                                             int ldb) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::rocsolver_zgetrs(handle,
                                trans,
                                n,
                                nrhs,
                                reinterpret_cast<rocblas_double_complex*>(a),
                                lda,
                                ipiv,
                                reinterpret_cast<rocblas_double_complex*>(b),
                                ldb));
}

#else
template <typename T>
void cusolver_getrs(const solverHandle_t& handle,
                    cublasOperation_t trans,
                    int n,
                    int nrhs,
                    T* a,
                    int lda,
                    int* ipiv,
                    T* b,
                    int ldb,
                    int* info);

template <>
void cusolver_getrs<float>(const solverHandle_t& handle,
                           cublasOperation_t trans,
                           int n,
                           int nrhs,
                           float* a,
                           int lda,
                           int* ipiv,
                           float* b,
                           int ldb,
                           int* info) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgetrs(
      handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info));
}

template <>
void cusolver_getrs<double>(const solverHandle_t& handle,
                            cublasOperation_t trans,
                            int n,
                            int nrhs,
                            double* a,
                            int lda,
                            int* ipiv,
                            double* b,
                            int ldb,
                            int* info) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgetrs(
      handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info));
}

template <>
void cusolver_getrs<dtype::complex<float>>(const solverHandle_t& handle,
                                           cublasOperation_t trans,
                                           int n,
                                           int nrhs,
                                           dtype::complex<float>* a,
                                           int lda,
                                           int* ipiv,
                                           dtype::complex<float>* b,
                                           int ldb,
                                           int* info) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCgetrs(handle,
                                trans,
                                n,
                                nrhs,
                                reinterpret_cast<cuComplex*>(a),
                                lda,
                                ipiv,
                                reinterpret_cast<cuComplex*>(b),
                                ldb,
                                info));
}

template <>
void cusolver_getrs<dtype::complex<double>>(const solverHandle_t& handle,
                                            cublasOperation_t trans,
                                            int n,
                                            int nrhs,
                                            dtype::complex<double>* a,
                                            int lda,
                                            int* ipiv,
                                            dtype::complex<double>* b,
                                            int ldb,
                                            int* info) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnZgetrs(handle,
                                trans,
                                n,
                                nrhs,
                                reinterpret_cast<cuDoubleComplex*>(a),
                                lda,
                                ipiv,
                                reinterpret_cast<cuDoubleComplex*>(b),
                                ldb,
                                info));
}

#endif  // PADDLE_WITH_HIP

template <typename T, typename Context>
void LuSolveKernel(const Context& dev_ctx,
                   const DenseTensor& b,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const std::string& trans,
                   DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  // Copy x to out since cusolverDn*getrs overwrites the input
  *out = phi::Transpose2DTo6D<Context, T>(dev_ctx, b);
  DenseTensor tem_lu = Transpose2DTo6D<Context, T>(dev_ctx, lu);
  // Validate input dimensions
  auto x_dims = b.dims();
  auto lu_dims = lu.dims();

#ifdef PADDLE_WITH_HIP
  rocblas_operation trans_op;
  if (trans == "N") {
    trans_op = rocblas_operation_none;
  } else if (trans == "T") {
    trans_op = rocblas_operation_transpose;
  } else if (trans == "C") {
    trans_op = rocblas_operation_conjugate_transpose;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "trans must be one of ['N', 'T', 'C'], but got %s", trans));
  }
#else
  cublasOperation_t trans_op;
  if (trans == "N") {
    trans_op = CUBLAS_OP_N;
  } else if (trans == "T") {
    trans_op = CUBLAS_OP_T;
  } else if (trans == "C") {
    trans_op = CUBLAS_OP_C;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "trans must be one of ['N', 'T', 'C'], but got %s", trans));
  }
#endif
  int n = static_cast<int>(lu_dims[lu_dims.size() - 1]);
  int nrhs = static_cast<int>(x_dims[x_dims.size() - 1]);
  int lda = std::max(1, n);
  int ldb = std::max(1, n);

  DenseTensor info_tensor;
  info_tensor.Resize({1});
  dev_ctx.template Alloc<int>(&info_tensor);
  int* d_info = info_tensor.data<int>();

  auto outdims = out->dims();
  auto outrank = outdims.size();
  int batchsize = product(common::slice_ddim(outdims, 0, outrank - 2));
  auto out_data = out->data<T>();
  auto lu_data = reinterpret_cast<T*>(const_cast<T*>(tem_lu.data<T>()));
  auto pivots_data =
      reinterpret_cast<int*>(const_cast<int*>(pivots.data<int>()));
  for (int i = 0; i < batchsize; i++) {
    auto handle = dev_ctx.cusolver_dn_handle();
    auto* out_data_item = &out_data[i * lda * nrhs];
    auto* lu_data_item = &lu_data[i * ldb * n];
    auto* pivots_data_item = &pivots_data[i * n];
#ifdef PADDLE_WITH_HIP
    rocsolver_getrs<T>(handle,
                       trans_op,
                       n,
                       nrhs,
                       lu_data_item,
                       lda,
                       pivots_data_item,
                       out_data_item,
                       ldb);
#else
    cusolver_getrs<T>(handle,
                      trans_op,
                      n,
                      nrhs,
                      lu_data_item,
                      lda,
                      pivots_data_item,
                      out_data_item,
                      ldb,
                      d_info);
#endif
  }
  *out = phi::Transpose2DTo6D<Context, T>(dev_ctx, *out);
}

}  // namespace phi

PD_REGISTER_KERNEL(lu_solve,
                   GPU,
                   ALL_LAYOUT,
                   phi::LuSolveKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
