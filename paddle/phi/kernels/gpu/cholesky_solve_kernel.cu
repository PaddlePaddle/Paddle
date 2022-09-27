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

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/impl/cholesky_solve_kernel_impl.h"

namespace phi {

template <typename T>
void cusolver_potrs(const solverHandle_t &handle,
                    cublasFillMode_t uplo,
                    int M,
                    int N,
                    T *Adata,
                    int lda,
                    T *Bdata,
                    int ldb,
                    int *devInfo);

template <>
void cusolver_potrs<float>(const solverHandle_t &handle,
                           cublasFillMode_t uplo,
                           int M,
                           int N,
                           float *Adata,
                           int lda,
                           float *Bdata,
                           int ldb,
                           int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSpotrs(
      handle, uplo, M, N, Adata, lda, Bdata, ldb, devInfo));
}

template <>
void cusolver_potrs<double>(const solverHandle_t &handle,
                            cublasFillMode_t uplo,
                            int M,
                            int N,
                            double *Adata,
                            int lda,
                            double *Bdata,
                            int ldb,
                            int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDpotrs(
      handle, uplo, M, N, Adata, lda, Bdata, ldb, devInfo));
}

template <>
void cusolver_potrs<phi::dtype::complex<float>>(
    const solverHandle_t &handle,
    cublasFillMode_t uplo,
    int M,
    int N,
    phi::dtype::complex<float> *Adata,
    int lda,
    phi::dtype::complex<float> *Bdata,
    int ldb,
    int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCpotrs(handle,
                                uplo,
                                M,
                                N,
                                reinterpret_cast<const cuComplex *>(Adata),
                                lda,
                                reinterpret_cast<cuComplex *>(Bdata),
                                ldb,
                                devInfo));
}

template <>
void cusolver_potrs<phi::dtype::complex<double>>(
    const cusolverDnHandle_t &handle,
    cublasFillMode_t uplo,
    int M,
    int N,
    phi::dtype::complex<double> *Adata,
    int lda,
    phi::dtype::complex<double> *Bdata,
    int ldb,
    int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZpotrs(
      handle,
      uplo,
      M,
      N,
      reinterpret_cast<const cuDoubleComplex *>(Adata),
      lda,
      reinterpret_cast<cuDoubleComplex *>(Bdata),
      ldb,
      devInfo));
}

template <typename T>
class CholeskySolveFunctor<T, GPUContext> {
 public:
  void operator()(const GPUContext &dev_ctx,
                  bool upper,
                  int M,
                  int N,
                  T *Adata,
                  int lda,
                  T *Bdata,
                  int *devInfo) {
    cublasFillMode_t uplo =
        upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    auto handle = dev_ctx.cusolver_dn_handle();
    cusolver_potrs<T>(handle, uplo, M, N, Adata, lda, Bdata, lda, devInfo);
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(cholesky_solve,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::CholeskySolveKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
