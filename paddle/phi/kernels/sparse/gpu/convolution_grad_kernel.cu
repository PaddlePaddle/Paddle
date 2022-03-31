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
                      const DenseTensor& out_grad,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      const int groups,
                      const bool subm,
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

  T* in_features_ptr = in_features.data<T>();
  T* d_x_features_ptr = d_x_features.data<T>();
  T* out_grad_features_ptr = out_grad_features.data<T>();
  kernel_grad->ResizeAndAllocate(kernel_dims);
  T* d_kernel_ptr = kernel_grad->data<T>();
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, kernel_grad, static_cast<T>(0.0f));

  int half_kernel_size = kernel_size / 2;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  x_grad->ResizeAndAllocate(x.non_zero_elements().dims());
  T* x_grad_values_ptr = x_grad->data<T>();
  set_zero(dev_ctx, x_grad, static_cast<T>(0.0f));
  set_zero(dev_ctx, &d_x_features, static_cast<T>(0.0f));

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
  int offset = 0, max_count = 0;
  for (int i = 0; i < kernel_size; i++) {
    offsets[i] = offset;
    offset += counter[i];
    if (i < half_kernel_size) {
      max_count = std::max(max_count, counter[i]);
    }
  }
  offsets[kernel_size] = offset;

  if (subm) {
    phi::funcs::sparse::SubmPreProcess<T, Context>(dev_ctx,
                                                   x,
                                                   kernel,
                                                   out_grad,
                                                   in_channels,
                                                   out_channels,
                                                   half_kernel_size,
                                                   kernel_grad,
                                                   x_grad);
    if (max_count == 0) {
      return;
    }
  }

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
                         dev_ctx.stream()>>>(out_grad.data<T>(),
                                             rulebook_ptr + rulebook_len * 2,
                                             out_grad_features_ptr,
                                             rulebook_len,
                                             out_channels);

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (counter[i] <= 0 || (subm && i == half_kernel_size)) {
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
  x_grad->ResizeAndAllocate(x.non_zero_elements().dims());
  DenseTensorMeta index_meta(DataType::INT32, {rulebook_len}, DataLayout::NCHW);
  DenseTensor out_index = phi::Empty(dev_ctx, std::move(index_meta));
  DenseTensor unique_key = phi::Empty(dev_ctx, std::move(index_meta));
  DenseTensor unique_value = phi::Empty(dev_ctx, std::move(index_meta));

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
                                         x_grad_values_ptr,
                                         subm);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_conv3d_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
