/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <thrust/device_vector.h>
#include <algorithm>
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/cholesky_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/matrix_rank_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"
#include "paddle/fluid/platform/for_range.h"

// #ifdef __NVCC__
// #include "cub/cub.cuh"
// #endif
// #ifdef __HIPCC__
// #include <hipcub/hipcub.hpp>
// namespace cub = hipcub;
// #endif

namespace paddle {
namespace operators {

// DDim A dynamically sized dimension.
// The number of dimensions must be between [1, 9].
using DDim = framework::DDim;
DDim UDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  // vectorize向量化
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 1] = k;
  return framework::make_ddim(x_vec);
}
DDim VHDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  return framework::make_ddim(x_vec);
}
DDim SDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  x_vec.erase(x_vec.end() - 1);  // rank - 1
  return framework::make_ddim(x_vec);
}

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
template <typename T>
__global__ void RankCount(T* data, int32_t* rank, float tol, int k) {
  int index = blockIdx.x * gridDim.x + threadIdx.x;
  if (index < k && data[index] > tol) {
    platform::CudaAtomicAdd(rank, 1);
  }
}

template <typename T>
__global__ void ComputeMax(T* data, T* max_data, int k) {
  int index = blockIdx.x * gridDim.x + threadIdx.x;
  if (index < k) {
    platform::CudaAtomicMax(max_data, data[index]);
  }
}

#endif

template <typename T>
class MatrixRankGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    // get in/output
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    auto* out_data = out->mutable_data<int32_t>(context.GetPlace());
    math::SetConstant<platform::CUDADeviceContext, int32_t> set_zero;
    set_zero(dev_ctx, out, static_cast<int32_t>(0));
    float tol = context.Attr<float>("tol");
    bool hermitian = context.Attr<bool>("hermitian");
    // get shape
    auto& dims = x->dims();
    int batch_count = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_count *= dims[i];
    }
    int dims_rank = dims.size();
    int m = dims[dims_rank - 2];
    int n = dims[dims_rank - 1];
    int k = std::min(m, n);

    // Must Copy X once, because the gesvdj will destory the content when exit.
    Tensor x_tmp;
    TensorCopy(*x, context.GetPlace(), &x_tmp);
    // cusolver API use
    auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_count);
    int* info_ptr = reinterpret_cast<int*>(info->ptr());

#if CUDA_VERSION >= 9020 && !defined(_WIN32)

    auto ComputeBlockSize = [](int col) {
      if (col > 512)
        return 1024;
      else if (col > 256 && col <= 512)
        return 512;
      else if (col > 128 && col <= 256)
        return 256;
      else if (col > 64 && col <= 128)
        return 128;
      else
        return 64;
    };

    if (hermitian) {
      // m == n
      VLOG(3) << "hermitian" << std::endl;
      Tensor W;
      auto* w_data = W.mutable_data<T>(framework::make_ddim({batch_count, m}),
                                       context.GetPlace());
      SyevjBatched(dev_ctx, batch_count, m, x_tmp.data<T>(), w_data, info_ptr);
      // compute abs(eigenvalues)
      Tensor W_ABS;
      auto* w_abs_data = W_ABS.mutable_data<T>(
          framework::make_ddim({batch_count, m}), context.GetPlace());
      platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                                W_ABS.numel());
      math::AbsFunctor<T> functor(w_data, w_abs_data, W_ABS.numel());
      for_range(functor);

      // std::vector<T> w_vec(W.numel());
      // TensorToVector(W, context.device_context(), &w_vec);
      // for (int i = 0; i < w_vec.size(); i++) {
      //   VLOG(3) << "w_vec: " << w_vec[i] << std::endl;
      // }
      // std::vector<T> w_abs_vec(W.numel());
      // TensorToVector(W_ABS, context.device_context(), &w_abs_vec);
      // for (int i = 0; i < w_abs_vec.size(); i++) {
      //   VLOG(3) << "w_abs_vec: " << w_abs_vec[i] << std::endl;
      // }

      int maxGridDimX = dev_ctx.GetCUDAMaxGridDimSize().x;
      // actually, int num_rows < max_grid_size
      int grid_size = m < maxGridDimX ? m : maxGridDimX;
      int block_size = ComputeBlockSize(n);

      // VLOG(3) << "grid_size: " << grid_size << std::endl;
      // VLOG(3) << "block_size: " << block_size << std::endl;

      auto cu_stream = dev_ctx.stream();
      for (int i = 0; i < batch_count; i++) {
        // compute tol
        Tensor max_tensor_gpu;
        auto* max_data_gpu = max_tensor_gpu.mutable_data<T>(
            framework::make_ddim({1}), context.GetPlace());
        ComputeMax<<<grid_size, block_size, 0, cu_stream>>>(w_abs_data + i * m,
                                                            max_data_gpu, m);
        Tensor max_tensor_cpu;
        TensorCopy(max_tensor_gpu, platform::CPUPlace(), &max_tensor_cpu);
        dev_ctx.Wait();
        VLOG(3) << "max_tensor: " << max_tensor_cpu.data<T>()[0] << std::endl;
        float tol_val = std::numeric_limits<float>::epsilon() * m *
                        max_tensor_cpu.data<T>()[0];
        tol_val = std::max(tol, tol_val);
        VLOG(3) << "tol_val: " << tol_val << std::endl;

        // std::vector<T> temp(max_tensor.numel());
        // TensorToVector(max_tensor, context.device_context(), &temp);
        // VLOG(3) << "max_tensor: " << temp[0] << std::endl;
        RankCount<<<grid_size, block_size, 0, cu_stream>>>(
            w_abs_data + i * m, out_data + i, tol_val, m);
      }
      // dev_ctx.Wait();

    } else {
      VLOG(3) << "not hermitian" << std::endl;

      Tensor U, VH, S;
      auto* vh_data = VH.mutable_data<T>(VHDDim(dims, k), context.GetPlace());
      auto* s_data = S.mutable_data<T>(SDDim(dims, k), context.GetPlace());
      auto* u_data = U.mutable_data<T>(UDDim(dims, k), context.GetPlace());
      GesvdjBatched(dev_ctx, batch_count, n, m, k, x_tmp.data<T>(), vh_data,
                    u_data, s_data, info_ptr, 1);

      // VLOG(3) << "tol: " << tol << std::endl;
      std::vector<T> s_vec(S.numel());
      TensorToVector(S, context.device_context(), &s_vec);
      for (int i = 0; i < s_vec.size(); i++) {
        VLOG(3) << "out vec: " << s_vec[i] << std::endl;
      }

      int maxGridDimX = dev_ctx.GetCUDAMaxGridDimSize().x;
      // actually, int num_rows < max_grid_size
      int grid_size = m < maxGridDimX ? m : maxGridDimX;
      int block_size = ComputeBlockSize(n);

      // VLOG(3) << "grid_size: " << grid_size << std::endl;
      // VLOG(3) << "block_size: " << block_size << std::endl;

      auto cu_stream = dev_ctx.stream();
      for (int i = 0; i < batch_count; i++) {
        // compute tol
        Tensor max_tensor_gpu;
        auto* max_data_gpu = max_tensor_gpu.mutable_data<T>(
            framework::make_ddim({1}), context.GetPlace());
        ComputeMax<<<grid_size, block_size, 0, cu_stream>>>(s_data + i * k,
                                                            max_data_gpu, k);
        Tensor max_tensor_cpu;
        TensorCopy(max_tensor_gpu, platform::CPUPlace(), &max_tensor_cpu);
        dev_ctx.Wait();
        VLOG(3) << "max_tensor: " << max_tensor_cpu.data<T>()[0] << std::endl;
        float tol_val = std::numeric_limits<float>::epsilon() * m *
                        max_tensor_cpu.data<T>()[0];
        tol_val = std::max(tol, tol_val);
        VLOG(3) << "tol_val: " << tol_val << std::endl;

        RankCount<<<grid_size, block_size, 0, cu_stream>>>(
            s_data + i * k, out_data + i, tol_val, k);
      }
      // dev_ctx.Wait();
    }

#endif
  }

  void GesvdjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                     int m, int n, int k, const T* cA, T* U, T* V, T* S,
                     int* info, int thin_UV = 1) const;

  void SyevjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                    int n, const T* cA, T* W, int* info) const;
};

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
template <>
void MatrixRankGPUKernel<float>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, const float* cA, float* U, float* V, float* S, int* info,
    int thin_UV) const {
  /* no compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_NOVECTOR; /* no compute singular vectors */
  gesvdjInfo_t gesvdj_params = NULL;
  float* A = const_cast<float*>(cA);
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSgesvdj_bufferSize(
      handle, jobz, thin_UV, m, n, A, lda, S, U, ldu, V, ldt, &lwork,
      gesvdj_params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSgesvdj(
        handle, jobz, thin_UV, m, n, A + stride_A * i, lda, S + k * i,
        U + stride_U * i, ldu, V + stride_V * i, ldt, workspace_ptr, lwork,
        info, gesvdj_params));
    // platform::dynload::cusolverDnSgesvdj(
    //         handle, jobz, thin_UV, m, n, A+stride_A*i, lda, S+k*i,
    //         U+stride_U*i,
    //         ldu, V+stride_V*i, ldt, workspace_ptr, lwork, info,
    //         gesvdj_params);
    // std::cout << "info:" << *info << std::endl;
    // check the error info
    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void MatrixRankGPUKernel<double>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, const double* cA, double* U, double* V, double* S, int* info,
    int thin_UV) const {
  /* no compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_NOVECTOR; /* no compute singular vectors */
  gesvdjInfo_t gesvdj_params = NULL;
  double* A = const_cast<double*>(cA);
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDgesvdj_bufferSize(
      handle, jobz, thin_UV, m, n, A, lda, S, U, ldu, V, ldt, &lwork,
      gesvdj_params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDgesvdj(
        handle, jobz, thin_UV, m, n, A + stride_A * i, lda, S + k * i,
        U + stride_U * i, ldu, V + stride_V * i, ldt, workspace_ptr, lwork,
        info, gesvdj_params));
    // check the error info
    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}
#endif

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
template <>
void MatrixRankGPUKernel<float>::SyevjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int n,
    const float* cA, float* W, int* info) const {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  // Lower triangle of A is stored
  // cusolver中矩阵是column-major即转置的形势，numpy和torch中使用下三角来进行计算
  // 因为转置的缘故需要上三角来进行计算
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  float* A = const_cast<float*>(cA);
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  // PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnXsyevjSetMaxSweeps(params,
  // 15));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSsyevj(
        handle, jobz, uplo, n, A + stride_A * i, lda, W + n * i, workspace_ptr,
        lwork, info, params));

    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroySyevjInfo(params));
}

template <>
void MatrixRankGPUKernel<double>::SyevjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int n,
    const double* cA, double* W, int* info) const {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  //  Lower triangle of A is stored
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  double* A = const_cast<double*>(cA);
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());

  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDsyevj(
        handle, jobz, uplo, n, A + stride_A * i, lda, W + n * i, workspace_ptr,
        lwork, info, params));
    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroySyevjInfo(params));
}
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(matrix_rank, ops::MatrixRankGPUKernel<float>,
                        ops::MatrixRankGPUKernel<double>);
#endif  // not PADDLE_WITH_HIP
