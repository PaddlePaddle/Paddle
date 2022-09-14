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

#include "paddle/phi/kernels/sparse/pool_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT = int>
__global__ void MaxPoolGradCudaKernel(const T* in_features_ptr,
                                      const T* out_features_ptr,
                                      const T* out_grad_ptr,
                                      const IntT* rulebook_ptr,
                                      const int n,
                                      const int rulebook_len,
                                      const int channels,
                                      T* x_grad_ptr) {
  phi::funcs::MaxPoolGrad<T> grad_functor;
  CUDA_KERNEL_LOOP_TYPE(i, n * channels, int64_t) {
    int real_i = i / channels;
    int c = i - real_i * channels;
    IntT in_i = rulebook_ptr[real_i];
    IntT out_i = rulebook_ptr[real_i + rulebook_len];
    grad_functor.compute(in_features_ptr[in_i * channels + c],
                         out_features_ptr[out_i * channels + c],
                         out_grad_ptr[out_i * channels + c],
                         1,
                         &x_grad_ptr[in_i * channels + c]);
  }
}

template <typename T, typename IntT = int>
void MaxPoolCooGradGPUKernel(const GPUContext& dev_ctx,
                             const SparseCooTensor& x,
                             const DenseTensor& rulebook,
                             const DenseTensor& counter,
                             const SparseCooTensor& out,
                             const SparseCooTensor& out_grad,
                             const std::vector<int>& kernel_sizes,
                             SparseCooTensor* x_grad) {
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  const int in_channels = x.dims()[4];
  int rulebook_len = rulebook.dims()[1];
  const IntT* rulebook_ptr = rulebook.data<IntT>();
  std::vector<int> offsets(kernel_size + 1);
  const int* counter_ptr = counter.data<int>();
  phi::funcs::sparse::PrefixSum(counter_ptr, &offsets[0], kernel_size);

  const T* in_features_ptr = x.values().data<T>();
  const T* out_features_ptr = out.values().data<T>();
  const T* out_grad_ptr = out_grad.values().data<T>();
  // TODO(zhangkaihuo): call phi::sparse::EmptyLike
  DenseTensor x_grad_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
  DenseTensor x_grad_values = phi::EmptyLike<T>(dev_ctx, x.values());
  x_grad->SetMember(x_grad_indices, x_grad_values, x.dims(), true);
  T* x_grad_ptr = x_grad_values.data<T>();
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  set_zero(dev_ctx, &x_grad_values, static_cast<T>(0.0f));
  phi::Copy<GPUContext>(
      dev_ctx, x.indices(), dev_ctx.GetPlace(), false, &x_grad_indices);

  for (int i = 0; i < kernel_size; i++) {
    if (counter_ptr[i] <= 0) {
      continue;
    }

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, counter_ptr[i] * in_channels, 1);
    MaxPoolGradCudaKernel<T, IntT>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(in_features_ptr,
                               out_features_ptr,
                               out_grad_ptr,
                               rulebook_ptr + offsets[i],
                               counter_ptr[i],
                               rulebook_len,
                               in_channels,
                               x_grad_ptr);
  }
}

template <typename T, typename Context>
void MaxPoolCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const DenseTensor& rulebook,
                          const DenseTensor& counter,
                          const SparseCooTensor& out,
                          const SparseCooTensor& out_grad,
                          const std::vector<int>& kernel_sizes,
                          SparseCooTensor* x_grad) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "MaxPoolCooGradGPUKernel", ([&] {
        MaxPoolCooGradGPUKernel<T, data_t>(
            dev_ctx, x, rulebook, counter, out, out_grad, kernel_sizes, x_grad);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(maxpool_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaxPoolCooGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
