/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"

#include "paddle/phi/kernels/sparse/sparse_pool_grad_kernel.h"

namespace phi {
namespace sparse {

template <typename T>
__global__ void MaxPoolGradCudaKernel(const T* in_features_ptr,
                                      const T* out_features_ptr,
                                      const T* out_grad_ptr,
                                      const int* rulebook_ptr,
                                      const int n,
                                      const int rulebook_len,
                                      const int channels,
                                      T* x_grad_ptr) {
  phi::funcs::MaxPoolGrad<T> grad_functor;
  CUDA_KERNEL_LOOP_TYPE(i, n * channels, int64_t) {
    int real_i = i / channels;
    int c = i - real_i * channels;
    int in_i = rulebook_ptr[real_i];
    int out_i = rulebook_ptr[real_i + rulebook_len];
    grad_functor.compute(in_features_ptr[in_i * channels + c],
                         out_features_ptr[out_i * channels + c],
                         out_grad_ptr[out_i * channels + c],
                         1,
                         &x_grad_ptr[in_i * channels + c]);
  }
}

template <typename T, typename Context>
void MaxPoolGradKernel(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const DenseTensor& rulebook,
                       const SparseCooTensor& out,
                       const DenseTensor& out_grad,
                       const std::vector<int>& kernel_sizes,
                       DenseTensor* x_grad) {
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  const int in_channels = x.dims()[4];
  int rulebook_len = rulebook.dims()[1];
  const int* rulebook_ptr = rulebook.data<int>();
  std::vector<int> offsets(kernel_size + 1), counter(kernel_size, 0),
      h_counter(kernel_size);
  phi::backends::gpu::GpuMemcpyAsync(&h_counter[0],
                                     rulebook_ptr,
                                     rulebook_len * sizeof(int),
#ifdef PADDLE_WITH_HIP
                                     hipMemcpyDeviceToHost,
#else
                                     cudaMemcpyDeviceToHost,
#endif

                                     dev_ctx.stream());
  dev_ctx.Wait();
  for (int i = 0; i < rulebook_len; i++) {
    counter[h_counter[i]] += 1;
  }
  phi::funcs::sparse::PrefixSum(&counter[0], &offsets[0], kernel_size);

  const T* in_features_ptr = x.non_zero_elements().data<T>();
  const T* out_features_ptr = out.non_zero_elements().data<T>();
  const T* out_grad_ptr = out_grad.data<T>();
  T* x_grad_ptr = x_grad->data<T>();
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, x_grad, static_cast<T>(0.0f));

  for (int i = 0; i < kernel_size; i++) {
    if (counter[i] <= 0) {
      continue;
    }

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, counter[i] * in_channels, 1);
    MaxPoolGradCudaKernel<T><<<config.block_per_grid.x,
                               config.thread_per_block.x,
                               0,
                               dev_ctx.stream()>>>(
        in_features_ptr,
        out_features_ptr,
        out_grad_ptr,
        rulebook_ptr + offsets[i] + rulebook_len,
        counter[i],
        rulebook_len,
        in_channels,
        x_grad_ptr);
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_maxpool_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaxPoolGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
