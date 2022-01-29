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

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/cholesky_solve_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
void cusolver_potrs(const cusolverDnHandle_t &cusolverH, cublasFillMode_t uplo,
                    int n, int nrhs, T *Adata, int lda, T *Bdata, int ldb,
                    int *devInfo);

template <>
void cusolver_potrs<float>(const cusolverDnHandle_t &cusolverH,
                           cublasFillMode_t uplo, int n, int nrhs, float *Adata,
                           int lda, float *Bdata, int ldb, int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSpotrs(
      cusolverH, uplo, n, nrhs, Adata, lda, Bdata, ldb, devInfo));
}

template <>
void cusolver_potrs<double>(const cusolverDnHandle_t &cusolverH,
                            cublasFillMode_t uplo, int n, int nrhs,
                            double *Adata, int lda, double *Bdata, int ldb,
                            int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDpotrs(
      cusolverH, uplo, n, nrhs, Adata, lda, Bdata, ldb, devInfo));
}

template <>
void cusolver_potrs<platform::complex<float>>(
    const cusolverDnHandle_t &cusolverH, cublasFillMode_t uplo, int n, int nrhs,
    platform::complex<float> *Adata, int lda, platform::complex<float> *Bdata,
    int ldb, int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnCpotrs(
      cusolverH, uplo, n, nrhs, reinterpret_cast<const cuComplex *>(Adata), lda,
      reinterpret_cast<cuComplex *>(Bdata), ldb, devInfo));
}

template <>
void cusolver_potrs<platform::complex<double>>(
    const cusolverDnHandle_t &cusolverH, cublasFillMode_t uplo, int n, int nrhs,
    platform::complex<double> *Adata, int lda, platform::complex<double> *Bdata,
    int ldb, int *devInfo) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnZpotrs(
      cusolverH, uplo, n, nrhs,
      reinterpret_cast<const cuDoubleComplex *>(Adata), lda,
      reinterpret_cast<cuDoubleComplex *>(Bdata), ldb, devInfo));
}

template <typename T>
class CholeskySolveFunctor<paddle::platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext &dev_ctx, bool upper, int n,
                  int nrhs, T *Adata, int lda, T *Bdata, int *devInfo) {
    cublasFillMode_t uplo =
        upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    /* step 1: get cusolver handle*/
    auto cusolverH = dev_ctx.cusolver_dn_handle();

    /* step 2: solve A0*X0 = B0  */
    cusolver_potrs<T>(cusolverH, uplo, n, nrhs, Adata, lda, Bdata, lda,
                      devInfo);

    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  }
};

template <typename T>
class MatrixReduceSumFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const Tensor &in, Tensor *out,
                  const framework::ExecutionContext &ctx) {
    // For example: in's dim = [5, 3, 2, 7, 3] ; out's dim = [3, 1, 7, 3]
    // out_reduce_dim should be [0, 2]
    const std::vector<std::int64_t> in_dims = framework::vectorize(in.dims());
    auto in_size = in_dims.size();
    const std::vector<std::int64_t> out_dims =
        framework::vectorize(out->dims());
    auto out_size = out_dims.size();

    std::vector<std::int64_t> out_bst_dims(in_size);

    std::fill(out_bst_dims.data(), out_bst_dims.data() + in_size - out_size, 1);
    std::copy(out_dims.data(), out_dims.data() + out_size,
              out_bst_dims.data() + in_size - out_size);

    std::vector<int> out_reduce_dims;
    for (size_t idx = 0; idx <= in_size - 3; idx++) {
      if (in_dims[idx] != 1 && out_bst_dims[idx] == 1) {
        out_reduce_dims.push_back(idx);
      }
    }
    gpuStream_t stream = ctx.cuda_device_context().stream();
    TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        in, out, kps::IdentityFunctor<T>(), out_reduce_dims, stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cholesky_solve,
    ops::CholeskySolveKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CholeskySolveKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    cholesky_solve_grad,
    ops::CholeskySolveGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CholeskySolveGradKernel<paddle::platform::CUDADeviceContext, double>);

#endif  // not PADDLE_WITH_HIP
