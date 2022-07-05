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

#include "paddle/phi/kernels/sparse/convolution_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/funcs/sparse/scatter.cu.h"
#include "paddle/phi/kernels/sparse/gpu/convolution.cu.h"

#include "glog/logging.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
void Conv3dGPUKernel(const GPUContext& dev_ctx,
                     const SparseCooTensor& x,
                     const DenseTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const int groups,
                     const bool subm,
                     const std::string& key,
                     SparseCooTensor* out,
                     DenseTensor* rulebook,
                     DenseTensor* counter) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)
  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  std::vector<int> kernel_sizes(kernel_dims.size());
  for (int i = 0; i < kernel_dims.size(); i++) {
    kernel_sizes[i] = kernel_dims[i];
  }

  std::vector<int> subm_paddings(paddings), subm_strides(strides);
  if (subm) {
    // the out shape of subm_conv is same as input shape
    // reset the padding=kernel_size/2 and strides=1
    phi::funcs::sparse::ResetSubmKernelSizeAndStrides(
        kernel.dims(), &subm_paddings, &subm_strides);
  }

  phi::funcs::sparse::GetOutShape(
      x_dims, kernel_sizes, subm_paddings, dilations, subm_strides, &out_dims);
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];
  std::vector<int> offsets(kernel_size + 1), h_counter(kernel_size);

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensorMeta counter_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  DenseTensorMeta offsets_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  DenseTensor counter_per_kernel = phi::Empty(dev_ctx, std::move(counter_meta));
  DenseTensor offsets_per_kernel = phi::Empty(dev_ctx, std::move(offsets_meta));
  DenseTensorMeta index_meta(DataType::INT32, {1}, DataLayout::NCHW);
  DenseTensor out_index = phi::Empty(dev_ctx, std::move(index_meta));
  DenseTensor unique_value = phi::Empty(dev_ctx, std::move(index_meta));

  VLOG(6) << "call SubmConv3D or Conv3D " << subm << " and the key is " << key;
  int n = 0;
  const IntT* rulebook_ptr = nullptr;
  bool need_product_rulebook = true;
  if (subm && !key.empty()) {
    const auto* table = x.table(key);
    if (table != nullptr) {
      need_product_rulebook = false;
      const DenseTensor& rulebook = table->first;
      rulebook_ptr = rulebook.data<IntT>();
      memcpy(h_counter.data(), table->second.data(), kernel_size * sizeof(int));
      out->SetTablePtr(x.GetTablePtr());

      n = rulebook.dims()[1];

      DenseTensor out_indices =
          phi::EmptyLike<IntT>(dev_ctx, x.non_zero_indices());
      DenseTensor out_values =
          phi::EmptyLike<T>(dev_ctx, x.non_zero_elements());
      phi::Copy(dev_ctx,
                x.non_zero_indices(),
                dev_ctx.GetPlace(),
                false,
                &out_indices);
      out->SetMember(out_indices, out_values, out_dims, true);
      IntT offset = 0;
      for (int i = 0; i < kernel_size; i++) {
        offsets[i] = offset;
        offset += h_counter[i];
      }
      offsets[kernel_size] = offset;
    }
  }
  if (need_product_rulebook) {
    DenseTensor tmp_rulebook;
    n = ProductRuleBook<T, GPUContext, IntT>(dev_ctx,
                                             x,
                                             kernel_sizes,
                                             subm_paddings,
                                             dilations,
                                             subm_strides,
                                             out_dims,
                                             subm,
                                             &tmp_rulebook,
                                             &counter_per_kernel,
                                             &offsets_per_kernel,
                                             &out_index,
                                             &unique_value,
                                             out,
                                             &h_counter,
                                             &offsets);
    rulebook_ptr = tmp_rulebook.data<IntT>();

    out->SetTablePtr(x.GetTablePtr());
    if (!key.empty()) {
      out->SetTable(key, std::make_pair(tmp_rulebook, h_counter));
    } else {
      *rulebook = tmp_rulebook;
      counter->Resize({kernel_size});
      int* counter_ptr = dev_ctx.template HostAlloc<int>(counter);
      memcpy(counter_ptr, h_counter.data(), h_counter.size() * sizeof(int));
    }
  }

  // 2. gather
  DenseTensorMeta in_features_meta(
      x.dtype(), {n, in_channels}, DataLayout::NCHW);
  DenseTensorMeta out_features_meta(
      x.dtype(), {n, out_channels}, DataLayout::NCHW);
  phi::DenseTensor in_features =
      phi::Empty(dev_ctx, std::move(in_features_meta));
  phi::DenseTensor out_features =
      phi::Empty(dev_ctx, std::move(out_features_meta));
  T* in_features_ptr = in_features.data<T>();
  T* out_features_ptr = out_features.data<T>();
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  set_zero(dev_ctx, &out_features, static_cast<T>(0.0f));

  const int VecSize = VecBytes / sizeof(T);
  if (in_channels % VecSize == 0) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, n * in_channels / VecSize, 1);
    GatherKernel<T, IntT, VecSize>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(x.non_zero_elements().data<T>(),
                               rulebook_ptr,
                               in_features_ptr,
                               n,
                               in_channels);
  } else {
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n * in_channels, 1);
    GatherKernel<T, IntT, 1>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(x.non_zero_elements().data<T>(),
                               rulebook_ptr,
                               in_features_ptr,
                               n,
                               in_channels);
  }

  // 3. call gemm for every werght
  auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
  auto* out_values = out->mutable_non_zero_elements();
  T* out_values_ptr = out_values->data<T>();
  set_zero(dev_ctx, out_values, static_cast<T>(0.0f));

  if (subm) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n, 1);
    unique_value.ResizeAndAllocate(
        {static_cast<int>(out->nnz() * kernel_size)});
    out_index.ResizeAndAllocate({static_cast<int>(n)});
    int* out_index_ptr = out_index.data<int>();
    int* unique_value_ptr = unique_value.data<int>();
    phi::backends::gpu::GpuMemsetAsync(
        out_index_ptr, 0, sizeof(int) * n, dev_ctx.stream());
    GroupIndexs<<<config.block_per_grid,
                  config.thread_per_block,
                  0,
                  dev_ctx.stream()>>>(
        n, kernel_size, rulebook_ptr + n, out_index_ptr, unique_value_ptr);
  }
  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (h_counter[i] <= 0) {
      continue;
    }

    // call gemm: (n, in_channels) * (in_channels, out_channels)
    const int M = h_counter[i];
    const int K = in_channels;
    const int N = out_channels;
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
    T* tmp_out_ptr = out_features_ptr + offsets[i] * out_channels;

    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              tmp_in_ptr,
              tmp_kernel_ptr,
              static_cast<T>(0),
              tmp_out_ptr);
  }

  // 4. scatter
  if (out_channels % VecSize == 0) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, out->nnz() * out_channels / VecSize, 1);
    phi::funcs::sparse::ScatterKernelV2<T, VecSize>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(out_features_ptr,
                               out_index.data<int>(),
                               unique_value.data<int>(),
                               out->nnz(),
                               kernel_size,
                               out_channels,
                               1,
                               out_values_ptr);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, out->nnz() * out_channels, 1);
    phi::funcs::sparse::ScatterKernelV2<T, 1>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(out_features_ptr,
                               out_index.data<int>(),
                               unique_value.data<int>(),
                               out->nnz(),
                               kernel_size,
                               out_channels,
                               1,
                               out_values_ptr);
  }
}

/**
 * x: the input SparseCooTensor, shape is (N, D, H, W, C)
 * kernel: the weight data, shape is (D, H, W, C, OC)
 * out: the output SparseCooTensor, shape is (N, D, H, W, OC)
 * rulebook: return rulebook if key is not vailed else return nullptr
 * counter: return counter if key is not vailed else return nullptr
 **/
template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const DenseTensor& kernel,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const int groups,
                  const bool subm,
                  const std::string& key,
                  SparseCooTensor* out,
                  DenseTensor* rulebook,
                  DenseTensor* counter) {
  PD_VISIT_INTEGRAL_TYPES(
      x.non_zero_indices().dtype(), "Conv3dGPUKernel", ([&] {
        Conv3dGPUKernel<T, data_t>(dev_ctx,
                                   x,
                                   kernel,
                                   paddings,
                                   dilations,
                                   strides,
                                   groups,
                                   subm,
                                   key,
                                   out,
                                   rulebook,
                                   counter);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_conv3d,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
