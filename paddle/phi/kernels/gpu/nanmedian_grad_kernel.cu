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

#include "paddle/phi/kernels/nanmedian_grad_kernel.h"

#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;
inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T>
__global__ void KernelNanmedianGrad(const T* x_ptr,
                                    const T* medians_ptr,
                                    const T* out_grad_ptr,
                                    T* x_grad_ptr,
                                    int64_t stride,
                                    int64_t numel) {
  auto zero = static_cast<T>(0);
  CUDA_KERNEL_LOOP(index, numel) {
    int64_t row = static_cast<int64_t>(index / stride);
    int64_t m_row = 2 * row;
    if (isnan(x_ptr[index]) || isnan(medians_ptr[m_row]) ||
        (x_ptr[index] != medians_ptr[m_row] &&
         x_ptr[index] != medians_ptr[m_row + 1])) {
      x_grad_ptr[index] = zero;
    } else {
      x_grad_ptr[index] = out_grad_ptr[row];
    }
  }
}

template <typename T, typename Context>
void NanmedianGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& medians,
                         const DenseTensor& out_grad,
                         DenseTensor* x_grad) {
  auto stream = dev_ctx.stream();
  const T* x_ptr = x.data<T>();
  const T* m_ptr = medians.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();
  T* x_grad_ptr = dev_ctx.template Alloc<T>(x_grad);

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];

  KernelNanmedianGrad<
      T><<<GET_BLOCKS(numel), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      x_ptr, m_ptr, out_grad_ptr, x_grad_ptr, stride, numel);
}

}  // namespace phi

PD_REGISTER_KERNEL(nanmedian_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::NanmedianGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
