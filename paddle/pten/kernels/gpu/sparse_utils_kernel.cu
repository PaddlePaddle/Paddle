/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/sparse.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/copy_kernel.h"
#include "paddle/pten/kernels/sparse_utils_kernel.h"

namespace pten {

template <typename T>
inline __device__ bool dev_is_zero(const T* data, const int64_t cols) {
  const T zero = static_cast<T>(0);
  for (int64_t i = 0; i < cols; i++) {
    if (data[i] != zero) {
      return false;
    }
  }
  return true;
}

template <typename T>
__global__ void kernel_get_non_zero_nums(const T* dense_data,
                                         const int rows,
                                         const int cols,
                                         int* non_zero_num,
                                         int* temp_indexs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int counter;
  if (threadIdx.x == 0) counter = 0;
  __syncthreads();

  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    int index = -1;
    if (!dev_is_zero(dense_data + i * cols, cols)) {
      atomicAdd(&counter, 1);
      index = i;
    }
    temp_indexs[i] = index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(non_zero_num, counter);
  }
}

template <typename T>
__global__ void kernel_get_values_and_calc_indices(const T* dense_data,
                                                   const int64_t sparse_dim,
                                                   const int64_t cols,
                                                   const int64_t* x_dims,
                                                   const int non_zero_num,
                                                   const int* indexs,
                                                   int64_t* indices,
                                                   T* sparse_data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int64_t sparse_index = indexs[i];
    int64_t x_index = sparse_index;
    for (int64_t j = sparse_dim - 1; j >= 0; j--) {
      indices[j * non_zero_num + i] = sparse_index % x_dims[j];
      sparse_index /= x_dims[j];
    }

    for (int j = 0; j < cols; j++) {
      sparse_data[i * cols + j] = dense_data[x_index * cols + j];
    }
  }
}

template <typename T, typename Context>
void DenseToSparseCooKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const int64_t sparse_dim,
                            SparseCooTensor* out) {
  const T* x_data = x.data<T>();
  const auto& x_dims = x.dims();
  auto dims_2d = flatten_to_2d(x_dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];
  auto nums_meta = pten::DenseTensorMeta(DataType::INT32,
                                         paddle::framework::make_ddim({1}),
                                         pten::DataLayout::NCHW);
  DenseTensor nums = pten::Empty<T, Context>(dev_ctx, std::move(nums_meta));
  auto x_dims_meta = pten::DenseTensorMeta(
      DataType::INT64,
      paddle::framework::make_ddim({static_cast<int64_t>(x_dims.size())}),
      pten::DataLayout::NCHW);
  DenseTensor d_x_dims =
      pten::Empty<T, Context>(dev_ctx, std::move(x_dims_meta));

  const auto place = dev_ctx.GetPlace();

  // 1. get numbers of non zero elements, and get the index of non zero elements
  int* nums_ptr = nums.mutable_data<int>(place);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(nums_ptr, 0, sizeof(int), dev_ctx.stream()));
  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, rows);
  auto temp_indexs_meta =
      pten::DenseTensorMeta(DataType::INT32,
                            paddle::framework::make_ddim({rows}),
                            pten::DataLayout::NCHW);
  DenseTensor temp_indexs =
      pten::Empty<T, Context>(dev_ctx, std::move(temp_indexs_meta));
  int* temp_indexs_ptr = temp_indexs.mutable_data<int>(place);
  kernel_get_non_zero_nums<<<config.block_per_grid,
                             config.thread_per_block,
                             0,
                             dev_ctx.stream()>>>(
      x_data, rows, cols, nums_ptr, temp_indexs_ptr);
  thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
                 temp_indexs_ptr,
                 temp_indexs_ptr + rows,
                 -1);

  // 2. copy non_zero_num to host, copy x_dims to device
  int non_zero_num = 0;
  paddle::memory::Copy(paddle::platform::CPUPlace(),
                       &non_zero_num,
                       paddle::platform::CUDAPlace(),
                       nums_ptr,
                       sizeof(int),
                       dev_ctx.stream());

  paddle::memory::Copy(paddle::platform::CUDAPlace(),
                       d_x_dims.mutable_data<int64_t>(place),
                       paddle::platform::CPUPlace(),
                       x_dims.Get(),
                       x_dims.size() * sizeof(x_dims[0]),
                       dev_ctx.stream());
  dev_ctx.Wait();

  const auto values_dims = InferDenseDims(x_dims, sparse_dim, non_zero_num);
  DenseTensorMeta indices_meta(
      DataType::INT64,
      paddle::framework::make_ddim(
          {sparse_dim, static_cast<int64_t>(non_zero_num)}),
      DataLayout::NCHW);
  DenseTensorMeta values_meta(x.meta().dtype, values_dims, x.meta().layout);
  pten::DenseTensor indices(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(indices_meta));
  pten::DenseTensor values(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(values_meta));
  int64_t* indices_data = indices.mutable_data<int64_t>(place);
  T* sparse_data = values.mutable_data<T>(place);

  // 3. calc indices by indexs and get values by indexs
  paddle::platform::GpuLaunchConfig config2 =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, non_zero_num);
  kernel_get_values_and_calc_indices<<<config2.block_per_grid,
                                       config2.thread_per_block,
                                       0,
                                       dev_ctx.stream()>>>(
      x_data,
      sparse_dim,
      cols,
      d_x_dims.data<int64_t>(),
      non_zero_num,
      temp_indexs_ptr,
      indices_data,
      sparse_data);
  out->SetMember(indices, values, x_dims, true);
}

template <typename ValueT, typename IndicesT>
__global__ void kernel_sparse_coo_to_dense(const IndicesT* indices,
                                           const IndicesT* sparse_offsets,
                                           const ValueT* data,
                                           ValueT* dense_data,
                                           const IndicesT non_zero_num,
                                           const int64_t base_offset,
                                           const int64_t sparse_dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += indices[j * non_zero_num + i] * sparse_offsets[j];
    }

    for (int j = 0; j < base_offset; j++) {
      dense_data[index * base_offset + j] = data[i * base_offset + j];
    }
  }
}

template <typename T, typename Context>
void SparseCooToDenseKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            DenseTensor* out) {
  const auto non_zero_num = x.nnz();
  const auto dense_dims = x.dims();
  const auto indices = x.non_zero_indices();
  const auto values = x.non_zero_elements();
  const auto indices_dims = indices.dims();
  int64_t sparse_dim = indices_dims[0];
  if (indices_dims.size() == 1) {
    sparse_dim = 1;
  }
  const int64_t dense_dim = values.dims().size() - 1;

  const auto place = dev_ctx.GetPlace();
  const T* x_data = values.data<T>();
  T* out_data = out->mutable_data<T>(place);
  int64_t base_offset = 1;
  for (int64_t i = 0; i < dense_dim; i++) {
    base_offset *= dense_dims[sparse_dim + i];
  }
  std::vector<int64_t> sparse_offsets(sparse_dim);
  int64_t offset = 1;
  for (int i = sparse_dim - 1; i >= 0; i--) {
    sparse_offsets[i] = offset;
    offset *= dense_dims[i];
  }

  auto sparse_offset_meta =
      pten::DenseTensorMeta(DataType::INT64,
                            paddle::framework::make_ddim({sparse_dim}),
                            pten::DataLayout::NCHW);
  DenseTensor d_sparse_offsets =
      pten::Empty<T, Context>(dev_ctx, std::move(sparse_offset_meta));
  paddle::memory::Copy(paddle::platform::CUDAPlace(),
                       d_sparse_offsets.mutable_data<int64_t>(place),
                       paddle::platform::CPUPlace(),
                       sparse_offsets.data(),
                       sparse_dim * sizeof(int64_t),
                       dev_ctx.stream());

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(out_data, 0, sizeof(T) * out->numel(), dev_ctx.stream()));
  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, non_zero_num);

  kernel_sparse_coo_to_dense<T, int64_t><<<config.block_per_grid,
                                           config.thread_per_block,
                                           0,
                                           dev_ctx.stream()>>>(
      indices.data<int64_t>(),
      d_sparse_offsets.data<int64_t>(),
      x_data,
      out_data,
      non_zero_num,
      base_offset,
      sparse_dim);
}

__global__ void kernel_get_batch_sizes(const int64_t* crows,
                                       const int m,
                                       const int batchs,
                                       int* batch_sizes) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < batchs) {
    batch_sizes[tid] = crows[tid * m];
  }
}

__global__ void kernel_convert_csr_crows_to_coo_rows(const int64_t* crows_ptr,
                                                     const int* crows_offsets,
                                                     int64_t* rows_ptr,
                                                     int64_t* batch_ptr,
                                                     const int rows) {
  const int b = blockIdx.y;
  const int64_t offset = crows_offsets ? crows_offsets[b] : 0;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    for (int j = crows_ptr[b * rows + i]; j < crows_ptr[b * rows + i + 1];
         j++) {
      rows_ptr[offset + j] = i;
      if (batch_ptr) {
        batch_ptr[offset + j] = b;
      }
    }
  }
}

template <typename T, typename Context>
void SparseCsrToCooKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          SparseCooTensor* out) {
  const DDim& x_dims = x.dims();
  const int64_t non_zero_num = x.non_zero_cols().numel();
  const auto& csr_crows = x.non_zero_crows();
  const auto& csr_cols = x.non_zero_cols();
  const auto& csr_values = x.non_zero_elements();
  const int64_t* csr_crows_data = csr_crows.data<int64_t>();
  const int64_t* csr_cols_data = csr_cols.data<int64_t>();
  const T* csr_values_data = csr_values.data<T>();

  int64_t sparse_dim = 2;
  if (x_dims.size() == 3) {
    sparse_dim = 3;
  }
  int batchs = x_dims.size() == 2 ? 1 : x_dims[0];
  int rows = x_dims.size() == 2 ? x_dims[0] : x_dims[1];

  const auto place = dev_ctx.GetPlace();
  DenseTensorMeta indices_meta(
      DataType::INT64, {sparse_dim, non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(x.dtype(), {non_zero_num}, x.layout());
  DenseTensorMeta offsets_meta(DataType::INT32, {batchs}, DataLayout::NCHW);
  DenseTensor indices =
      pten::Empty<int64_t, Context>(dev_ctx, std::move(indices_meta));
  DenseTensor values = pten::Empty<T, Context>(dev_ctx, std::move(values_meta));
  DenseTensor offsets =
      pten::Empty<T, Context>(dev_ctx, std::move(offsets_meta));
  int64_t* coo_indices = indices.mutable_data<int64_t>(place);
  int64_t* batch_ptr = x_dims.size() == 2 ? nullptr : coo_indices;
  int64_t* coo_rows_data =
      x_dims.size() == 2 ? coo_indices : batch_ptr + non_zero_num;
  int64_t* coo_cols_data = coo_rows_data + non_zero_num;
  int* offsets_ptr = batchs == 1 ? nullptr : offsets.mutable_data<int>(place);
  T* coo_values_data = values.mutable_data<T>(place);

  if (batchs > 1) {
    paddle::platform::GpuLaunchConfig config1 =
        paddle::platform::GetGpuLaunchConfig1D(dev_ctx, batchs);
    kernel_get_batch_sizes<<<config1.block_per_grid,
                             config1.thread_per_block>>>(
        csr_crows_data, rows, batchs, offsets_ptr);

    thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
                           offsets_ptr,
                           offsets_ptr + batchs,
                           offsets_ptr);
  }

  paddle::platform::GpuLaunchConfig config2 =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, rows);
  config2.block_per_grid.y = batchs;
  kernel_convert_csr_crows_to_coo_rows<<<config2.block_per_grid,
                                         config2.thread_per_block>>>(
      csr_crows_data, offsets_ptr, coo_rows_data, batch_ptr, rows);
  paddle::memory::Copy(place,
                       coo_cols_data,
                       place,
                       csr_cols_data,
                       sizeof(int64_t) * non_zero_num,
                       dev_ctx.stream());
  paddle::memory::Copy(place,
                       coo_values_data,
                       place,
                       csr_values_data,
                       sizeof(T) * non_zero_num,
                       dev_ctx.stream());
  out->SetMember(indices, values, x_dims, true);
}

__global__ void kernel_get_batchs_offset(const int64_t* batchs_ptr,
                                         const int non_zero_num,
                                         int64_t* batchs_offset) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    if (i == non_zero_num - 1 || batchs_ptr[i] != batchs_ptr[i + 1]) {
      batchs_offset[batchs_ptr[i]] = i + 1;
    }
  }
}

__global__ void kernel_convert_coo_rows_to_csr_crows(
    const int64_t* batchs_offset,  // can be null if batchs = 1
    const int64_t* coo_rows_data,
    int64_t* csr_crows_data,
    const int rows,
    const int64_t non_zero_num) {
  const int b = blockIdx.y;
  int batch_non_zero_num =
      batchs_offset == nullptr ? non_zero_num : batchs_offset[b];
  if (batch_non_zero_num == 0) return;
  int batch_start = 0;
  if (b > 0) {
    batch_start = batchs_offset[b - 1];
    batch_non_zero_num -= batch_start;
    if (threadIdx.x == 0) printf("batch_start = %d\n", batch_start);
  }
  auto* coo_rows_ptr = coo_rows_data + batch_start;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < batch_non_zero_num; i += gridDim.x * blockDim.x) {
    if (i == 0) {
      for (int j = 0; j <= coo_rows_ptr[0]; j++) {
        csr_crows_data[b * (rows + 1) + j] = 0;
      }
    } else {
      for (int j = coo_rows_ptr[i - 1]; j < coo_rows_ptr[i]; j++) {
        csr_crows_data[b * (rows + 1) + j + 1] = i;
      }
    }
    if (i == batch_non_zero_num - 1) {
      for (int64_t i = coo_rows_ptr[batch_non_zero_num - 1] + 1; i < rows + 1;
           i++) {
        csr_crows_data[b * (rows + 1) + i] = batch_non_zero_num;
      }
    }
  }
}

template <typename T, typename Context>
void SparseCooToCsrKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          SparseCsrTensor* out) {
  const auto& x_dims = x.dims();
  bool valid = x_dims.size() == 2 || x_dims.size() == 3;
  PADDLE_ENFORCE_EQ(valid,
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D or 3-D matrix"));
  const int64_t non_zero_num = x.nnz();
  if (non_zero_num <= 0) return;

  int batchs = x_dims.size() == 2 ? 1 : x_dims[0];
  int rows = x_dims.size() == 2 ? x_dims[0] : x_dims[1];

  const auto place = dev_ctx.GetPlace();
  DenseTensorMeta crows_meta(
      DataType::INT64, {batchs * (rows + 1)}, DataLayout::NCHW);
  DenseTensorMeta cols_meta(DataType::INT64, {non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(x.dtype(), {non_zero_num}, x.layout());
  pten::DenseTensor non_zero_crows(
      pten::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(crows_meta));
  pten::DenseTensor non_zero_cols(
      pten::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(cols_meta));
  pten::DenseTensor non_zero_elements(
      pten::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(values_meta));
  int64_t* csr_crows_data = non_zero_crows.mutable_data<int64_t>(place);
  int64_t* csr_cols_data = non_zero_cols.mutable_data<int64_t>(place);
  T* csr_values_data = non_zero_elements.mutable_data<T>(place);

  const auto& coo_indices = x.non_zero_indices();
  const auto& coo_values = x.non_zero_elements();
  const int64_t* batchs_ptr = coo_indices.data<int64_t>();
  const int64_t* coo_rows_data =
      batchs == 1 ? batchs_ptr : batchs_ptr + non_zero_num;
  const int64_t* coo_cols_data = coo_rows_data + non_zero_num;
  const T* coo_values_data = coo_values.data<T>();

  if (!x.coalesced()) {
    // TODO(zhangkahuo): call coalesced() to distinct and sort the indices
  }
  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, non_zero_num);
  if (batchs > 1) {
    DenseTensorMeta batchs_meta(DataType::INT64, {batchs}, DataLayout::NCHW);
    pten::DenseTensor batchs_offset(
        pten::make_intrusive<paddle::experimental::SharedStorage>(place),
        std::move(batchs_meta));
    int64_t* batchs_offset_ptr = batchs_offset.mutable_data<int64_t>(place);
    kernel_get_batchs_offset<<<config.block_per_grid,
                               config.thread_per_block,
                               0,
                               dev_ctx.stream()>>>(
        batchs_ptr, non_zero_num, batchs_offset_ptr);
    config.block_per_grid.y = batchs;
    kernel_convert_coo_rows_to_csr_crows<<<config.block_per_grid,
                                           config.thread_per_block,
                                           0,
                                           dev_ctx.stream()>>>(
        batchs_offset_ptr, coo_rows_data, csr_crows_data, rows, non_zero_num);
    std::vector<int64_t> tmp(batchs * (rows + 1));
    cudaMemcpy(tmp.data(),
               csr_crows_data,
               sizeof(int64_t) * batchs * (rows + 1),
               cudaMemcpyDeviceToHost);
  } else {
    kernel_convert_coo_rows_to_csr_crows<<<config.block_per_grid,
                                           config.thread_per_block,
                                           0,
                                           dev_ctx.stream()>>>(
        nullptr, coo_rows_data, csr_crows_data, rows, non_zero_num);
  }
  paddle::memory::Copy(place,
                       csr_cols_data,
                       place,
                       coo_cols_data,
                       sizeof(int64_t) * non_zero_num,
                       dev_ctx.stream());
  paddle::memory::Copy(place,
                       csr_values_data,
                       place,
                       coo_values_data,
                       sizeof(T) * non_zero_num,
                       dev_ctx.stream());
  out->SetMember(non_zero_crows, non_zero_cols, non_zero_elements, x_dims);
}

}  // namespace pten

PT_REGISTER_KERNEL(dense_to_sparse_coo,
                   GPU,
                   ALL_LAYOUT,
                   pten::DenseToSparseCooKernel,
                   float,
                   double) {}

PT_REGISTER_KERNEL(sparse_csr_to_coo,
                   GPU,
                   ALL_LAYOUT,
                   pten::SparseCsrToCooKernel,
                   float,
                   double) {}

PT_REGISTER_KERNEL(sparse_coo_to_csr,
                   GPU,
                   ALL_LAYOUT,
                   pten::SparseCooToCsrKernel,
                   float,
                   double,
                   pten::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL(sparse_coo_to_dense,
                   GPU,
                   ALL_LAYOUT,
                   pten::SparseCooToDenseKernel,
                   float,
                   double) {}
#endif
