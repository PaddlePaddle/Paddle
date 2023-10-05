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

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/tril_triu_kernel.h"

namespace phi {

#define FUNC_WITH_TYPES(m) m(float, S) m(double, D)

#define POTRI_INSTANCE(T, C)                                             \
  void Potri(const GPUContext& dev_ctx,                                  \
             cublasFillMode_t uplo,                                      \
             int n,                                                      \
             T* A,                                                       \
             int lda,                                                    \
             int* info) {                                                \
    auto handle = dev_ctx.cusolver_dn_handle();                          \
    int workspace_size = 0;                                              \
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDn##C##potri_bufferSize( \
        handle, uplo, n, A, lda, &workspace_size));                      \
    auto workspace = phi::memory_utils::Alloc(                           \
        dev_ctx.GetPlace(),                                              \
        workspace_size * sizeof(T),                                      \
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))); \
    T* workspace_ptr = reinterpret_cast<T*>(workspace->ptr());           \
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDn##C##potri(            \
        handle, uplo, n, A, lda, workspace_ptr, workspace_size, info));  \
  }

FUNC_WITH_TYPES(POTRI_INSTANCE);

template <typename T, typename Context>
void CholeskyInverseKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           bool upper,
                           DenseTensor* out) {
  auto& dims = x.dims();
  int batch_count = 1;
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count *= static_cast<int>(dims[i]);
  }
  int m = dims[dims.size() - 1];

  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  DenseTensor trans_result = phi::TransposeLast2Dim<T>(dev_ctx, x);
  phi::Copy<Context>(dev_ctx, trans_result, dev_ctx.GetPlace(), false, out);
  auto* out_data = out->data<T>();

  DenseTensor info = phi::Empty<int, Context>(dev_ctx, IntArray({batch_count}));
  int* info_data = info.data<int>();

  cublasFillMode_t uplo =
      upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  for (int i = 0; i < batch_count; i++) {
    Potri(
        dev_ctx, uplo, m, out_data + i * m * m, std::max(1, m), info_data + i);
  }

  std::vector<int> error_info;
  error_info.resize(batch_count);
  memory_utils::Copy(CPUPlace(),
                     error_info.data(),
                     dev_ctx.GetPlace(),
                     info_data,
                     sizeof(int) * batch_count,
                     dev_ctx.stream());

  for (int i = 0; i < batch_count; i++) {
    PADDLE_ENFORCE_EQ(
        error_info[i],
        0,
        errors::InvalidArgument("Cholesky inverse was not successful. The "
                                "%d-th input matrice "
                                "might not be invertible.",
                                i));
  }
  DenseTensor* tri = new DenseTensor();
  tri->Resize(x.dims());
  dev_ctx.template Alloc<T>(tri);

  if (upper) {
    phi::TrilTriuKernel<T>(dev_ctx, *out, -1, true, tri);
  } else {
    phi::TrilTriuKernel<T>(dev_ctx, *out, 1, false, tri);
  }
  trans_result = phi::TransposeLast2Dim<T>(dev_ctx, *tri);
  phi::AddKernel<T>(dev_ctx, *out, trans_result, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(cholesky_inverse,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::CholeskyInverseKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
