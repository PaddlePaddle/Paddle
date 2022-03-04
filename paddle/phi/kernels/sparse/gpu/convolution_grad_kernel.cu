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
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/sparse/convolution_grad_kernel.h"
#include "paddle/phi/kernels/sparse/gpu/convolution.cu.h"

namespace phi {
namespace sparse {

// rulebook[3, rulebook_len]:
//[
//  [kernel_index],
//  [in_i],
//  [out_i],
//]
// x_grad = out_grad * transpose(kenrel)
// kernel_grad = transpose(x) * out_grad
template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const DenseTensor& rulebook,
                      const DenseTensor& kernel,
                      const SparseCooTensor& out_grad,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      const int groups,
                      DenseTensor* x_grad,
                      DenseTensor* kernel_grad) {
  const auto& kernel_dims = kernel.dims();
  const int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];
  const int* rulebook_ptr = rulebook.data<int>();

  const int rulebook_len = rulebook.dims()[1];

  DenseTensorMeta in_features_meta(
      x.dtype(), {rulebook_len, in_channels}, DataLayout::NCHW);
  DenseTensorMeta d_x_features_meta(
      x.dtype(), {rulebook_len, in_channels}, DataLayout::NCHW);
  DenseTensorMeta out_grad_features_meta(
      x.dtype(), {rulebook_len, out_channels}, DataLayout::NCHW);
  phi::DenseTensor in_features =
      phi::Empty(dev_ctx, std::move(in_features_meta));
  phi::DenseTensor d_x_features =
      phi::Empty(dev_ctx, std::move(d_x_features_meta));
  phi::DenseTensor out_grad_features =
      phi::Empty(dev_ctx, std::move(out_grad_features_meta));

  dev_ctx.Alloc(
      &in_features, in_features.dtype(), sizeof(T) * in_features.numel());
  T* in_features_ptr = in_features.data<T>();
  dev_ctx.Alloc(
      &d_x_features, d_x_features.dtype(), sizeof(T) * d_x_features.numel());
  T* d_x_features_ptr = d_x_features.data<T>();
  dev_ctx.Alloc(&out_grad_features,
                out_grad_features.dtype(),
                sizeof(T) * out_grad_features.numel());
  T* out_grad_features_ptr = out_grad_features.data<T>();
  kernel_grad->Resize(kernel_dims);
  dev_ctx.Alloc(
      kernel_grad, kernel_grad->dtype(), kernel_grad->numel() * sizeof(T));
  T* d_kernel_ptr = kernel_grad->data<T>();
  phi::funcs::SetConstant<Context, int> set_zero;
  set_zero(dev_ctx, kernel_grad, 0);

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, rulebook_len * in_channels, 1);
  GatherKernel<T, int><<<config.block_per_grid.x,
                         config.thread_per_block.x,
                         0,
                         dev_ctx.stream()>>>(x.non_zero_elements().data<T>(),
                                             rulebook_ptr + rulebook_len,
                                             in_features_ptr,
                                             rulebook_len,
                                             in_channels);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, rulebook_len * out_channels, 1);
  GatherKernel<T, int><<<config.block_per_grid.x,
                         config.thread_per_block.x,
                         0,
                         dev_ctx.stream()>>>(
      out_grad.non_zero_elements().data<T>(),
      rulebook_ptr + rulebook_len * 2,
      out_grad_features_ptr,
      rulebook_len,
      out_channels);

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  std::vector<int> offsets(kernel_size + 1), counter(kernel_size, 0),
      h_counter(rulebook_len, 0);
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
  int offset = 0;
  for (int i = 0; i < kernel_size; i++) {
    offsets[i] = offset;
    offset += counter[i];
  }
  offsets[kernel_size] = offset;

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (counter[i] <= 0) {
      continue;
    }

    const int M = counter[i];
    const int K = in_channels;
    const int N = out_channels;
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    T* tmp_out_grad_ptr = out_grad_features_ptr + offsets[i] * out_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * in_channels * out_channels;
    T* tmp_d_x_ptr = d_x_features_ptr + offsets[i] * out_channels;
    T* tmp_d_kernel_ptr = d_kernel_ptr + i * in_channels * out_channels;

    // call gemm: d_kernel = transpose(x) * out_grad
    // (in_channels, n) * (n, out_channels)
    blas.GEMM(CblasTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              tmp_in_ptr,
              tmp_out_grad_ptr,
              static_cast<T>(0),
              tmp_d_kernel_ptr);

    // call gemm: d_x = out_grad * transpose(kernel)
    // (n, out_channels) * (out_channels, in_channels)
    blas.GEMM(CblasNoTrans,
              CblasTrans,
              M,
              K,
              N,
              static_cast<T>(1),
              tmp_out_grad_ptr,
              tmp_kernel_ptr,
              static_cast<T>(0),
              tmp_d_x_ptr);
  }

  // 4. scatter
  x_grad->Resize(x.non_zero_elements().dims());
  dev_ctx.Alloc(x_grad, x_grad->dtype(), sizeof(T) * x_grad->numel());
  T* x_grad_values_ptr = x_grad->data<T>();

  DenseTensor out_index = phi::Empty<int, Context>(dev_ctx);
  DenseTensor unique_key = phi::Empty<int, Context>(dev_ctx);
  DenseTensor unique_value = phi::Empty<int, Context>(dev_ctx);
  unique_key.ResizeAndAllocate({rulebook_len});
  out_index.ResizeAndAllocate({rulebook_len});
  unique_value.ResizeAndAllocate({rulebook_len});
  dev_ctx.Alloc(
      &unique_key, unique_key.dtype(), sizeof(int) * unique_key.numel());
  dev_ctx.Alloc(&out_index, out_index.dtype(), sizeof(int) * out_index.numel());
  dev_ctx.Alloc(
      &unique_value, unique_value.dtype(), sizeof(int) * unique_value.numel());

  SortedAndUniqueIndex(dev_ctx,
                       rulebook_ptr + rulebook_len,
                       rulebook_len,
                       &out_index,
                       &unique_key,
                       &unique_value);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, rulebook_len * in_channels, 1);

  ScatterKernel<T><<<config.block_per_grid.x,
                     config.thread_per_block.x,
                     0,
                     dev_ctx.stream()>>>(d_x_features_ptr,
                                         unique_value.data<int>(),
                                         out_index.data<int>(),
                                         x.nnz(),
                                         rulebook_len,
                                         in_channels,
                                         x_grad_values_ptr);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_conv3d_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(3).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
