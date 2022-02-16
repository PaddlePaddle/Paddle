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
#include "paddle/fluid/platform/dynload/cusolver.h"

namespace paddle {
namespace operators {

template <typename T>
class CholeskyGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    bool upper = context.Attr<bool>("upper");
    auto& dims = x->dims();
    int batch_count = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_count *= dims[i];
    }
    int m = dims[dims.size() - 1];
    int tensor_size = batch_count * m * m;

    const auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());

    // matrices are assumed to be stored in column-major order in cusolver
    cublasFillMode_t uplo =
        upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // portf is inplace, thus copy the triangular part of the input matrices to
    // the output and set the other triangular part to 0 firstly
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                              tensor_size);
    if (upper) {
      MatrixBandPartFunctor<T> matrix_band_part_functor(
          m, m, /* num_lower_diags */ 0, /* num_upper_diags */ m, x_data,
          out_data);
      for_range(matrix_band_part_functor);
    } else {
      MatrixBandPartFunctor<T> matrix_band_part_functor(
          m, m, /* num_lower_diags */ m, /* num_upper_diags */ 0, x_data,
          out_data);
      for_range(matrix_band_part_functor);
    }

    auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_count);
    auto* info_ptr = reinterpret_cast<int*>(info->ptr());

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
    if (batch_count > 1) {
      std::vector<T*> output_ptrs;
      for (int i = 0; i < batch_count; i++) {
        output_ptrs.emplace_back(out_data + i * m * m);
      }
      thrust::device_vector<T*> dev_output_ptrs(output_ptrs.begin(),
                                                output_ptrs.end());
      PotrfBatched(dev_ctx, uplo, m,
                   thrust::raw_pointer_cast(dev_output_ptrs.data()), m,
                   info_ptr, batch_count);
      // TODO(guosheng): There seems to a bug in cusolver potrfBatched and need
      // to clear the upper triangle of the output. Remove this workaround once
      // the bug is fixed.
      if (!upper) {
        MatrixBandPartFunctor<T> matrix_band_part_functor(
            m, m, /* num_lower_diags */ m, /* num_upper_diags */ 0, out_data,
            out_data);
        for_range(matrix_band_part_functor);
      }
    } else {
#endif
      for (int i = 0; i < batch_count; i++) {
        Potrf(dev_ctx, uplo, m, out_data + i * m * m, m, info_ptr + i);
      }

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
    }
#endif
    // check the info
    std::vector<int> error_info;  // only for checking positive matrix
    error_info.resize(batch_count);

    memory::Copy(platform::CPUPlace(), error_info.data(), dev_ctx.GetPlace(),
                 info_ptr, sizeof(int) * batch_count, dev_ctx.stream());

    for (int i = 0; i < batch_count; ++i) {
      PADDLE_ENFORCE_EQ(error_info[i], 0,
                        platform::errors::PreconditionNotMet(
                            "For batch [%d]: U(%d, %d) is zero, singular U.", i,
                            error_info[i], error_info[i]));
    }
  }

  void Potrf(const platform::CUDADeviceContext& dev_ctx, cublasFillMode_t uplo,
             int n, T* A, int lda, int* info) const;

  void PotrfBatched(const platform::CUDADeviceContext& dev_ctx,
                    cublasFillMode_t uplo, int n, T* Aarray[], int lda,
                    int* info_array, int batch_size) const;
};

#define FUNC_WITH_TYPES(m) m(float, S) m(double, D)

#define POTRF_INSTANCE(T, C)                                                   \
  template <>                                                                  \
  void CholeskyGPUKernel<T>::Potrf(const platform::CUDADeviceContext& dev_ctx, \
                                   cublasFillMode_t uplo, int n, T* A,         \
                                   int lda, int* info) const {                 \
    auto handle = dev_ctx.cusolver_dn_handle();                                \
    int workspace_size = 0;                                                    \
    PADDLE_ENFORCE_GPU_SUCCESS(                                                \
        platform::dynload::cusolverDn##C##potrf_bufferSize(                    \
            handle, uplo, n, A, lda, &workspace_size));                        \
    auto workspace = memory::Alloc(dev_ctx, workspace_size);                   \
    T* workspace_ptr = reinterpret_cast<T*>(workspace->ptr());                 \
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDn##C##potrf(        \
        handle, uplo, n, A, lda, workspace_ptr, workspace_size, info));        \
  }

FUNC_WITH_TYPES(POTRF_INSTANCE);

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
#define POTRF_BATCH_INSTANCE(T, C)                                             \
  template <>                                                                  \
  void CholeskyGPUKernel<T>::PotrfBatched(                                     \
      const platform::CUDADeviceContext& dev_ctx, cublasFillMode_t uplo,       \
      int n, T* Aarray[], int lda, int* info_array, int batch_size) const {    \
    auto handle = dev_ctx.cusolver_dn_handle();                                \
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDn##C##potrfBatched( \
        handle, uplo, n, Aarray, lda, info_array, batch_size));                \
  }

FUNC_WITH_TYPES(POTRF_BATCH_INSTANCE);
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cholesky, ops::CholeskyGPUKernel<float>,
                        ops::CholeskyGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    cholesky_grad,
    ops::CholeskyGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CholeskyGradKernel<paddle::platform::CUDADeviceContext, double>);

#endif  // not PADDLE_WITH_HIP
