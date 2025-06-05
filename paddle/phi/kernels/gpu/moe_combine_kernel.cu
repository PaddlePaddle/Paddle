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

#include "paddle/phi/kernels/moe_combine_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T>
__global__ void combine_moe_kernel(const T* x,
                                   const T* combine_weights,
                                   const int* scatter_index,
                                   T* y,
                                   const int64_t k,
                                   const int64_t seqlen,
                                   const int64_t hidden_size,
                                   const int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    int64_t row_i = i / hidden_size;
    int64_t slice_i = i - row_i * hidden_size;
    const int* scatter_index_start = scatter_index + row_i * k;
    T* dest_ptr = y + i;
    for (int ki = 0; ki < k; ki++) {
      // get combine_weights i
      const T* w_ptr = combine_weights + row_i * k + ki;
      const T* x_ptr =
          x + static_cast<int64_t>(*(scatter_index_start + ki)) * hidden_size +
          slice_i;
      *(dest_ptr) += (*w_ptr) * (*x_ptr);
    }
  }
}

template <typename T>
void combine_moe_kernelLauncher(const T* x,
                                const T* combine_weights,
                                const int* scatter_index,
                                T* y,
                                const int64_t k,
                                const int64_t seqlen,
                                const int64_t hidden_size,
                                cudaStream_t stream) {
  // y is [seqlen, hidden_size]
  // for kk in k:
  //     y[i][j] += x[scatter_index[i][kk]][j] * combine_weights[i][kk]
  const int64_t n = hidden_size * seqlen;

  const int64_t threads = 1024;
  const int64_t blocks = (n + threads - 1) / threads;

  combine_moe_kernel<T><<<blocks, threads, 0, stream>>>(
      x, combine_weights, scatter_index, y, k, seqlen, hidden_size, n);
}

template <typename T>
void apply_moe_combine_fwd(const T* x,
                           const T* combine_weights,
                           const int* scatter_index,
                           T* y,
                           const int64_t k,
                           const int64_t seqlen,
                           const int64_t hidden_size,
                           cudaStream_t stream) {
  combine_moe_kernelLauncher<T>(
      x, combine_weights, scatter_index, y, k, seqlen, hidden_size, stream);
}

template <typename T, typename Context>
void moe_combine_fwd(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& combine_weights,
                     const DenseTensor& scatter_index,
                     const DenseTensor& y,
                     const int64_t k,
                     const int64_t seqlen,
                     const int64_t hidden_size) {
  apply_moe_combine_fwd<T>(x.data<T>(),
                           combine_weights.data<T>(),
                           scatter_index.data<int>(),
                           const_cast<T*>(y.data<T>()),
                           k,
                           seqlen,
                           hidden_size,
                           dev_ctx.stream());
}

template <typename T, typename Context>
void MoeCombineKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& combine_weights,
                      const DenseTensor& scatter_index,
                      DenseTensor* y) {
  dev_ctx.template Alloc<T>(y);  // T cannot support phi::dtype::float8 very
                                 // well, maybe replaced with x.dtype();
  phi::Full<T, Context>(
      dev_ctx, phi::IntArray(common::vectorize(y->dims())), 0, y);
  auto combine_weights_shape = combine_weights.dims();
  auto x_shape = x.dims();
  moe_combine_fwd<T, Context>(dev_ctx,
                              x,
                              combine_weights,
                              scatter_index,
                              *y,
                              combine_weights_shape[1],  // k
                              combine_weights_shape[0],  // seqlen
                              x_shape[1]);               // hidden_size
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_combine,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeCombineKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
