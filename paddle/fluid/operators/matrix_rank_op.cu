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
#include "paddle/fluid/operators/elementwise/svd_eigen.h"
#include "paddle/fluid/operators/matrix_rank_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

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

template <typename T>
class MatrixRankGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    float tol = context.Attr<float>("tol");
    bool hermitian = context.Attr<bool>("hermitian");

    auto& dims = x->dims();
    int batch_count = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_count *= dims[i];
    }
    int dims_rank = dims.size();
    int m = dims[dims_rank - 2];
    int n = dims[dims_rank - 1];
    int k = std::min(m, n);

    Tensor U, VH, S;
    auto* vh_data = VH.mutable_data<T>(VHDDim(dims, k), context.GetPlace());
    auto* s_data = S.mutable_data<T>(SDDim(dims, k), context.GetPlace());
    auto* u_data = U.mutable_data<T>(UDDim(dims, k), context.GetPlace());

    // VLOG(3) << "S numel: " << S.numel() << std::endl;

    // std::vector<T> vec(S.numel());
    // TensorToVector(S, context.device_context(), &vec);
    // for(int i=0; i< static_cast<int>(vec.size()); ++i){
    //   VLOG(3) << "vec[" << i<< "] : "<< vec[i];
    // }

    // std::cout << U.dims()[0] << "  " << U.dims()[1] << std::endl ;
    // std::cout << VH.dims()[0] << "  " << VH.dims()[1] << std::endl ;
    // std::cout << S.dims()[0] << std::endl ;

    auto* out_data = out->mutable_data<math::Real<int32_t>>(context.GetPlace());

    // Must Copy X once, because the gesvdj will destory the content when exit.
    Tensor x_tmp;
    TensorCopy(*x, context.GetPlace(), &x_tmp);
    // VLOG(3) << "x_temp: " << *x_tmp.data<T>() << std::endl;

    auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_count);
    int* info_ptr = reinterpret_cast<int*>(info->ptr());

#if CUDA_VERSION >= 9020 && !defined(_WIN32)

    if (hermitian) {
      std::cout << "hermitian" << std::endl;

    } else {
      // std::cout << "not hermitian" << std::endl;
      GesvdjBatched(dev_ctx, batch_count, n, m, k, x_tmp.data<T>(), vh_data,
                    u_data, s_data, info_ptr, 1);
      dev_ctx.Wait();

      // auto dito =
      // math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
      // T>(context);
      // dito.print_matrix(S, "S");

      // std::cout << S.dims() << std::endl;
      // T* data = S.data<T>();
      // std::cout << *(data) << std::endl;

      // VLOG(3) << "+++++++++++++++" << std::endl;
      // VLOG(3) << std::to_string( *(s_data) ) << std::endl;

      // std::vector<T> vec(S.numel());
      // TensorToVector(S, context.device_context(), &vec);
      // for(int i=0; i< static_cast<int>(vec.size()); ++i){
      //   VLOG(3) << "vec[" << i<< "] : "<< vec[i];
      // }
      // std::cout << S.dims();

      VLOG(3) << "----------" << s_data[0] << std::endl;

      std::vector<T> s_vec(S.numel());
      TensorToVector(S, context.device_context(), &s_vec);
      std::vector<int32_t> out_vec(out->numel());

      for (int i = 0; i < batch_count; i++) {
        int rank = 0;
        for (int j = 0; j < k; j++) {
          if (s_vec[i * k + j] > tol) {
            rank = rank + 1;
          }
        }
        out_vec[i] = rank;
      }
      TensorFromVector(out_vec, context.device_context(), out);
    }

#endif
  }

  void GesvdjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                     int m, int n, int k, const T* A, T* U, T* V, T* S,
                     int* info, int thin_UV = 1) const;
};

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
template <>
void MatrixRankGPUKernel<float>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, const float* cA, float* U, float* V, float* S, int* info,
    int thin_UV) const {
  /* no compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; /* no compute singular vectors */
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
#endif

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
template <>
void MatrixRankGPUKernel<double>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, const double* cA, double* U, double* V, double* S, int* info,
    int thin_UV) const {
  /* no compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; /* no compute singular vectors */
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(matrix_rank, ops::MatrixRankGPUKernel<float>,
                        ops::MatrixRankGPUKernel<double>);
#endif  // not PADDLE_WITH_HIP