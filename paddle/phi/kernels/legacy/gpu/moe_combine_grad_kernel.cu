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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
namespace phi {

template <typename T>
__global__ void combine_moe_bwd_kernel(const T* x,
                                       const T* combine_weights,
                                       const int* scatter_index,
                                       const T* grad_y,
                                       T* grad_x,
                                       T* grad_combine_weights_helper,
                                       const int64_t k,
                                       const int64_t seqlen,
                                       const int64_t hidden_size,
                                       const int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    int64_t row_i = i / hidden_size;
    int64_t slice_i = i - row_i * hidden_size;
    const int* scatter_index_start = scatter_index + row_i * k;
    const T grad_y_i = *(grad_y + i);
    // y [ row_i, slice_i]
    // combine [row_i, k, slice_i]
    int64_t weight_base = row_i * k * hidden_size + slice_i;

    T* grad_cw_ptr =
        grad_combine_weights_helper + weight_base;  // stride hidden_size
    for (int64_t ki = 0; ki < k; ki++) {
      // get combine_weights i
      int64_t ele_index =
          static_cast<int64_t>(*(scatter_index_start + ki)) * hidden_size +
          slice_i;
      const T* w_ptr = combine_weights + row_i * k + ki;
      const T* x_ptr = x + ele_index;
      if ((*w_ptr) != T(0)) {
        *(grad_x + ele_index) = grad_y_i * (*w_ptr);
      }
      *(grad_cw_ptr + ki * hidden_size) = grad_y_i * (*x_ptr);
    }
  }
}

template <typename T>
void combine_moe_bwd_kernelLauncher(const T* x,
                                    const T* combine_weights,
                                    const int* scatter_index,
                                    const T* grad_y,
                                    T* grad_x,
                                    T* grad_combine_weights_helper,
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

  combine_moe_bwd_kernel<T>
      <<<blocks, threads, 0, stream>>>(x,
                                       combine_weights,
                                       scatter_index,
                                       grad_y,
                                       grad_x,
                                       grad_combine_weights_helper,
                                       k,
                                       seqlen,
                                       hidden_size,
                                       n);
}

template <typename T>
void apply_moe_combine_bwd(const T* x,
                           const T* combine_weights,
                           const int* scatter_index,
                           const T* grad_y,
                           T* grad_x,
                           T* grad_combine_weights_helper,
                           const int64_t k,
                           const int64_t seqlen,
                           const int64_t hidden_size,
                           cudaStream_t stream) {
  combine_moe_bwd_kernelLauncher<T>(x,
                                    combine_weights,
                                    scatter_index,
                                    grad_y,
                                    grad_x,
                                    grad_combine_weights_helper,
                                    k,
                                    seqlen,
                                    hidden_size,
                                    stream);
}

template <typename T, typename Context>
void moe_combine_bwd(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& combine_weights,
                     const DenseTensor& scatter_index,
                     const DenseTensor& grad_y,
                     const DenseTensor* grad_x,
                     const DenseTensor* grad_combine_weights_helper,
                     const int64_t k,
                     const int64_t seqlen,
                     const int64_t hidden_size) {
  apply_moe_combine_bwd<T>(
      x.data<T>(),
      combine_weights.data<T>(),
      scatter_index.data<int>(),
      grad_y.data<T>(),
      const_cast<T*>(grad_x->data<T>()),
      const_cast<T*>(grad_combine_weights_helper->data<T>()),
      k,
      seqlen,
      hidden_size,
      dev_ctx.stream());
}
template <typename T, typename Context>
void MoeCombineGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& combine_weights,
                          const DenseTensor& scatter_index,
                          const DenseTensor& grad_y,
                          DenseTensor* grad_x,
                          DenseTensor* grad_combine_weights_helper) {
  dev_ctx.template Alloc<T>(grad_x);
  dev_ctx.template Alloc<T>(grad_combine_weights_helper);
  phi::Full<T, Context>(
      dev_ctx, phi::IntArray(common::vectorize(grad_x->dims())), 0, grad_x);
  phi::Full<T, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(grad_combine_weights_helper->dims())),
      0,
      grad_combine_weights_helper);
  auto x_shape = x.dims();
  auto combine_weights_shape = combine_weights.dims();
  moe_combine_bwd<T, Context>(dev_ctx,
                              x,
                              combine_weights,
                              scatter_index,
                              grad_y,
                              grad_x,
                              grad_combine_weights_helper,
                              combine_weights_shape[1],  // k
                              combine_weights_shape[0],  // seqlen
                              x_shape[1]);               // hidden_size
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_combine_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeCombineGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
