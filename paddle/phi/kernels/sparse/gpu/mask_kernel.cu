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

#include "paddle/phi/kernels/sparse/mask_kernel.h"

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.cu.h"
#include "paddle/phi/kernels/funcs/sparse/utils.cu.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
__global__ void MaskKernel(const T* x_ptr,
                           const IntT* indices_ptr,
                           const int64_t* sparse_offsets,
                           const int64_t non_zero_num,
                           const int cols,
                           const int sparse_dim,
                           T* out_values_ptr) {
  CUDA_KERNEL_LOOP_TYPE(i, non_zero_num * cols, int64_t) {
    int64_t out_i = i / cols;
    int64_t col_i = i - out_i * cols;
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += indices_ptr[j * non_zero_num + out_i] * sparse_offsets[j];
    }
    out_values_ptr[out_i * cols + col_i] = x_ptr[index * cols + col_i];
  }
}

template <typename T, typename IntT>
void SparseMaskGPUKernel(const GPUContext& dev_ctx,
                         const DenseTensor& x,
                         const SparseCooTensor& mask,
                         SparseCooTensor* out) {
  const DDim& dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x.dims(),
      mask.dims(),
      phi::errors::InvalidArgument("the input x and mask must have the shape"));
  const DenseTensor& indices = mask.indices();
  const DenseTensor& values = mask.values();
  const int sparse_dim = mask.sparse_dim();
  DenseTensor sparse_offsets = phi::Empty<GPUContext>(
      dev_ctx,
      DenseTensorMeta(DataType::INT64, {sparse_dim}, DataLayout::NCHW));
  std::vector<int64_t> h_sparse_offsets(sparse_dim);
  phi::funcs::sparse::CalcOffsetsPerDim(
      dims, sparse_dim, h_sparse_offsets.data());

  phi::backends::gpu::GpuMemcpyAsync(sparse_offsets.data<int64_t>(),
                                     &h_sparse_offsets[0],
                                     sizeof(int64_t) * sparse_dim,
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  DenseTensor out_indices = phi::EmptyLike<T>(dev_ctx, indices);
  DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, values);

  phi::Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &out_indices);

  const IntT* indices_ptr = indices.data<IntT>();
  T* out_values_ptr = out_values.data<T>();
  const T* x_ptr = x.data<T>();
  const int64_t non_zero_num = mask.nnz();
  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int cols = dims_2d[1];

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num * cols, 1);
  MaskKernel<T, IntT>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          x_ptr,
          indices_ptr,
          sparse_offsets.data<int64_t>(),
          non_zero_num,
          cols,
          sparse_dim,
          out_values_ptr);

  out->SetMember(out_indices, out_values, dims, true);
}

/**
 * @brief Filter the DenseTensor x by the
 * mask.indices() and output a SparseCooTensor
 * x and mask must have the same shape.
 **/
template <typename T, typename Context>
void SparseMaskKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const SparseCooTensor& mask,
                      SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      mask.indices().dtype(), "SparseMaskGPUKernel", ([&] {
        SparseMaskGPUKernel<T, data_t>(dev_ctx, x, mask, out);
      }));
}

template <typename IntT>
__global__ void MaskTable(const IntT* x_indexs,
                          const int n,
                          int* index_flags,
                          int* table) {
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    int index = x_indexs[i];
    phi::funcs::sparse::SetBits(index, index_flags);
    table[index] = i;
  }
}

template <typename T, typename IntT, int VecSize>
__global__ void MaskCopy(const IntT* mask_indexs,
                         const int* index_flags,
                         const int* table,
                         const int n,
                         const int stride,
                         const T* x_values,
                         T* out_values) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    const int mask_index = mask_indexs[i];
    const bool flag = phi::funcs::sparse::TestBits(mask_index, index_flags);
    if (flag) {
      int j = table[mask_index];
      for (int k = 0; k < stride; k += VecSize) {
        LoadT vec_x;
        phi::Load<T, VecSize>(x_values + j * stride + k, &vec_x);
        phi::Store<T, VecSize>(vec_x, out_values + i * stride + k);
      }
    }
  }
}

template <typename T, typename IntT>
void SparseMaskHelperGPUKernel(const GPUContext& dev_ctx,
                               const SparseCooTensor& x,
                               const DenseTensor& mask_indices,
                               DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      mask_indices.dims().size(),
      2,
      phi::errors::InvalidArgument("the mask_indices must be 2-D tensor"));

  const int32_t sparse_dim = x.sparse_dim();
  auto indices_dtype = paddle::experimental::CppTypeToDataType<IntT>::Type();

  std::vector<IntT> sparse_offsets(sparse_dim);

  DenseTensorMeta x_indexs_meta(indices_dtype, {x.nnz()}, DataLayout::NCHW);
  DenseTensorMeta mask_indexs_meta(
      indices_dtype, {mask_indices.dims()[1]}, DataLayout::NCHW);
  DenseTensorMeta sparse_offset_meta(
      indices_dtype, {sparse_dim}, DataLayout::NCHW);

  DenseTensor x_indexs =
      phi::Empty<GPUContext>(dev_ctx, std::move(x_indexs_meta));
  DenseTensor mask_indexs =
      phi::Empty<GPUContext>(dev_ctx, std::move(mask_indexs_meta));
  DenseTensor bound_out =
      phi::Empty<GPUContext>(dev_ctx, std::move(mask_indexs_meta));
  DenseTensor d_sparse_offsets =
      phi::Empty<GPUContext>(dev_ctx, std::move(sparse_offset_meta));
  IntT* x_indexs_ptr = x_indexs.data<IntT>();
  IntT* mask_indexs_ptr = mask_indexs.data<IntT>();
  IntT* bound_out_ptr = bound_out.data<IntT>();

  // 1. calc the offsets of per dim
  phi::funcs::sparse::CalcOffsetsPerDim(
      x.dims(), sparse_dim, sparse_offsets.data());
  // 2. copy sparse_offsets to device
  phi::backends::gpu::GpuMemcpyAsync(d_sparse_offsets.data<IntT>(),
                                     sparse_offsets.data(),
                                     sizeof(IntT) * sparse_dim,
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  // 3. flatten x indices and mask indices
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_indexs.numel(), 1);
  phi::funcs::sparse::FlattenIndicesKernel<<<config.block_per_grid,
                                             config.thread_per_block,
                                             0,
                                             dev_ctx.stream()>>>(
      x.indices().data<IntT>(),
      d_sparse_offsets.data<IntT>(),
      x_indexs.numel(),
      sparse_dim,
      x_indexs_ptr);

  config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, mask_indexs.numel(), 1);
  phi::funcs::sparse::FlattenIndicesKernel<<<config.block_per_grid,
                                             config.thread_per_block,
                                             0,
                                             dev_ctx.stream()>>>(
      mask_indices.data<IntT>(),
      d_sparse_offsets.data<IntT>(),
      mask_indexs.numel(),
      sparse_dim,
      mask_indexs_ptr);

  int table_size = 1;
  auto x_dims = x.dims();
  for (int i = 0; i < sparse_dim; i++) {
    table_size *= x_dims[i];
  }
  DenseTensor table = phi::Empty<int>(dev_ctx, {table_size});
  DenseTensor index_flags = phi::Empty<int>(dev_ctx, {(table_size + 31) / 32});
  phi::backends::gpu::GpuMemsetAsync(index_flags.data<int>(),
                                     0,
                                     index_flags.numel() * sizeof(int),
                                     dev_ctx.stream());
  const int64_t stride =
      x.dims().size() == sparse_dim ? 1 : x.values().dims()[1];
  *out = phi::EmptyLike<T>(dev_ctx, x.values());
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  set_zero(dev_ctx, out, static_cast<T>(0));
  T* out_ptr = out->data<T>();
  config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_indexs.numel(), 1);
  MaskTable<<<config.block_per_grid,
              config.thread_per_block,
              0,
              dev_ctx.stream()>>>(x_indexs_ptr,
                                  x_indexs.numel(),
                                  index_flags.data<int>(),
                                  table.data<int>());
  config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, mask_indexs.numel(), 1);

  const int VecBytes = 16;
  const int VecSize = VecBytes / sizeof(T);
  if (stride % VecSize == 0) {
    MaskCopy<T, IntT, VecSize><<<config.block_per_grid,
                                 config.thread_per_block,
                                 0,
                                 dev_ctx.stream()>>>(mask_indexs_ptr,
                                                     index_flags.data<int>(),
                                                     table.data<int>(),
                                                     mask_indexs.numel(),
                                                     stride,
                                                     x.values().data<T>(),
                                                     out_ptr);
  } else {
    MaskCopy<T, IntT, 1><<<config.block_per_grid,
                           config.thread_per_block,
                           0,
                           dev_ctx.stream()>>>(mask_indexs_ptr,
                                               index_flags.data<int>(),
                                               table.data<int>(),
                                               mask_indexs.numel(),
                                               stride,
                                               x.values().data<T>(),
                                               out_ptr);
  }
}

template <typename T, typename Context>
void SparseMaskHelperKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& mask_indices,
                            DenseTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "SparseMaskHelperGPUKernel", ([&] {
        SparseMaskHelperGPUKernel<T, data_t>(dev_ctx, x, mask_indices, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(mask,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseMaskKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(mask_helper,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseMaskHelperKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
