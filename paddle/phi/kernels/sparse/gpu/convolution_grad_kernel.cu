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

#include "paddle/phi/kernels/sparse/convolution_grad_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/funcs/sparse/scatter.cu.h"
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
template <typename T, typename IntT>
void Conv3dGradGPUKernel(const GPUContext& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& kernel,
                         const SparseCooTensor& out,
                         const SparseCooTensor& out_grad,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         const int groups,
                         const bool subm,
                         const std::string& key,
                         SparseCooTensor* x_grad,
                         DenseTensor* kernel_grad) {
  const auto& kernel_dims = kernel.dims();
  const int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  const auto* table = out.table(key);
  const DenseTensor& rulebook = table->first;
  const IntT* rulebook_ptr = rulebook.data<IntT>();

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
  *kernel_grad = phi::EmptyLike<T>(dev_ctx, kernel);
  T* d_kernel_ptr = kernel_grad->data<T>();
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  // set_zero(dev_ctx, kernel_grad, static_cast<T>(0.0f));
  phi::backends::gpu::GpuMemsetAsync(
      d_kernel_ptr, 0, sizeof(T) * kernel_grad->numel(), dev_ctx.stream());

  int half_kernel_size = kernel_size / 2;
  auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
  DenseTensor x_grad_indices =
      phi::EmptyLike<IntT>(dev_ctx, x.non_zero_indices());
  DenseTensor x_grad_values = phi::EmptyLike<T>(dev_ctx, x.non_zero_elements());
  T* x_grad_values_ptr = x_grad_values.data<T>();
  // set_zero(dev_ctx, &x_grad_values, static_cast<T>(0.0f));
  phi::backends::gpu::GpuMemsetAsync(x_grad_values_ptr,
                                     0,
                                     sizeof(T) * x_grad_values.numel(),
                                     dev_ctx.stream());
  // set_zero(dev_ctx, &d_x_features, static_cast<T>(0.0f));
  phi::backends::gpu::GpuMemsetAsync(
      d_x_features_ptr, 0, sizeof(T) * d_x_features.numel(), dev_ctx.stream());
  phi::Copy<GPUContext>(dev_ctx,
                        x.non_zero_indices(),
                        dev_ctx.GetPlace(),
                        false,
                        &x_grad_indices);
  x_grad->SetMember(x_grad_indices, x_grad_values, x.dims(), true);

  std::vector<int> offsets(kernel_size + 1);
  const auto& counter = table->second;

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
    phi::funcs::sparse::SubmPreProcess<T, GPUContext>(
        dev_ctx,
        x,
        kernel,
        out_grad.non_zero_elements(),
        in_channels,
        out_channels,
        half_kernel_size,
        kernel_grad,
        &x_grad_values);
    if (max_count == 0) {
      return;
    }
  }

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
  DenseTensor unique_value = phi::Empty<int>(
      dev_ctx, {static_cast<int>(x_grad->nnz() * kernel_size * 2)});
  DenseTensor out_index =
      phi::Empty<int>(dev_ctx, {static_cast<int>(x.nnz() * 2)});
  int* out_index_ptr = out_index.data<int>();
  int* unique_value_ptr = unique_value.data<int>();
  cudaMemsetAsync(
      out_index_ptr, 0, sizeof(int) * x.nnz() * 2, dev_ctx.stream());

  GroupIndexsV2<<<config.block_per_grid,
                  config.thread_per_block,
                  0,
                  dev_ctx.stream()>>>(rulebook_len,
                                      x.nnz(),
                                      kernel_size,
                                      offsets[kernel_size / 2],
                                      rulebook_ptr,
                                      out_index_ptr,
                                      unique_value_ptr);

  const int VecSize = VecBytes / sizeof(T);
  if (in_channels % VecSize == 0) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, x.nnz() * in_channels / VecSize, 1);
    GatherKernelV3<T, IntT, VecSize>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(x.non_zero_elements().data<T>(),
                               out_index_ptr,
                               unique_value_ptr,
                               x.nnz(),
                               kernel_size,
                               in_features_ptr,
                               in_channels);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, x.nnz() * in_channels, 1);
    GatherKernelV3<T, IntT, 1>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(x.non_zero_elements().data<T>(),
                               out_index_ptr,
                               unique_value_ptr,
                               x.nnz(),
                               kernel_size,
                               in_features_ptr,
                               in_channels);
  }

  if (out_channels % VecSize == 0) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, rulebook_len * out_channels / VecSize, 1);
    GatherKernel<T, IntT, VecSize>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(out_grad.non_zero_elements().data<T>(),
                               rulebook_ptr + rulebook_len,
                               out_grad_features_ptr,
                               rulebook_len,
                               out_channels);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, rulebook_len * out_channels, 1);
    GatherKernel<T, IntT, 1>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(out_grad.non_zero_elements().data<T>(),
                               rulebook_ptr + rulebook_len,
                               out_grad_features_ptr,
                               rulebook_len,
                               out_channels);
  }

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
    T* tmp_d_x_ptr = d_x_features_ptr + offsets[i] * in_channels;
    T* tmp_d_kernel_ptr = d_kernel_ptr + i * in_channels * out_channels;

    // call gemm: d_kernel = transpose(x) * out_grad
    // (in_channels, n) * (n, out_channels)
    blas.GEMM(CblasTrans,
              CblasNoTrans,
              K,
              N,
              M,
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
  if (in_channels % VecSize == 0) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, x_grad->nnz() * in_channels / VecSize, 1);
    phi::funcs::sparse::ScatterKernelV3<T, VecSize>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(d_x_features_ptr,
                               out_index.data<int>(),
                               unique_value.data<int>(),
                               x_grad->nnz(),
                               kernel_size,
                               in_channels,
                               x_grad_values_ptr);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, x_grad->nnz() * in_channels, 1);
    phi::funcs::sparse::ScatterKernelV3<T, 1>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(d_x_features_ptr,
                               out_index.data<int>(),
                               unique_value.data<int>(),
                               x_grad->nnz(),
                               kernel_size,
                               in_channels,
                               x_grad_values_ptr);
  }
}

template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const DenseTensor& kernel,
                      const SparseCooTensor& out,
                      const SparseCooTensor& out_grad,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      const int groups,
                      const bool subm,
                      const std::string& key,
                      SparseCooTensor* x_grad,
                      DenseTensor* kernel_grad) {
  PD_VISIT_INTEGRAL_TYPES(
      x.non_zero_indices().dtype(), "Conv3dGradGPUKernel", ([&] {
        Conv3dGradGPUKernel<T, data_t>(dev_ctx,
                                       x,
                                       kernel,
                                       out,
                                       out_grad,
                                       paddings,
                                       dilations,
                                       strides,
                                       groups,
                                       subm,
                                       key,
                                       x_grad,
                                       kernel_grad);
      }));
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
