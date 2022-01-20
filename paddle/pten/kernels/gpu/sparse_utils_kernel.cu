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

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

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
                                         int* non_zero_num) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int counter;
  if (threadIdx.x == 0) counter = 0;
  __syncthreads();

  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    if (!dev_is_zero(dense_data + i * cols, cols)) {
      atomicAdd(&counter, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(non_zero_num, counter);
  }
}

template <typename T>
__global__ void kernel_get_non_zero_indexs(const T* dense_data,
                                           const int rows,
                                           const int cols,
                                           int* non_zero_num,
                                           int64_t* indexs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int counter, block_start;
  extern __shared__ int64_t cache[];
  if (threadIdx.x == 0) {
    counter = 0;
    block_start = 0;
  }
  __syncthreads();
  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    if (!dev_is_zero(dense_data + i * cols, cols)) {
      auto index = atomicAdd(&counter, 1);
      cache[index] = i;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    block_start = atomicAdd(non_zero_num, counter);
  }
  __syncthreads();
  if (threadIdx.x < counter) {
    indexs[block_start + threadIdx.x] = cache[threadIdx.x];
  }
}

template <typename T>
__global__ void kernel_get_values_and_calc_indices(const T* dense_data,
                                                   const int64_t sparse_dim,
                                                   const int64_t cols,
                                                   const int64_t* x_dims,
                                                   const int non_zero_num,
                                                   const int64_t* indexs,
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
  if (x_dims.size() == 2 && sparse_dim == 2) {
#if CUDA_VERSION >= 11020
    auto sparse =
        paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                           T>(dev_ctx);
    auto nnz_dims = paddle::framework::make_ddim({rows + 1});
    DenseTensorMeta nnz_meta(DataType::INT32, nnz_dims, DataLayout::NCHW);
    DenseTensor nnz_tensor =
        pten::Empty<int, Context>(dev_ctx, std::move(nnz_meta));
    int* nnz_ptr = nnz_tensor.mutable_data<int>();
    sparse.nnz(rows, cols, x_data, nnz_ptr, nnz_ptr + 1);
    int non_zero_num = 0;
    paddle::memory::Copy(paddle::platform::CPUPlace(),
                         &non_zero_num,
                         paddle::platform::CUDAPlace(),
                         nnz_ptr,
                         sizeof(int),
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

    int64_t* rows_ptr = indices.mutable_data<int64_t>();
    int64_t* cols_ptr = rows_ptr + non_zero_num;
    T* values_ptr = values.mutable_data<T>();
    sparse.DenseToSparseCoo(rows, cols, x_data, rows_ptr, cols_ptr, values_ptr);
    out->SetMember(indices, values, x_dims, true);
    return;
#endif
  }

  auto nums_meta = pten::DenseTensorMeta(DataType::INT32,
                                         paddle::framework::make_ddim({1}),
                                         pten::DataLayout::NCHW);
  DenseTensor nums = pten::Empty<T, Context>(dev_ctx, std::move(nums_meta));
  auto x_dims_meta = pten::DenseTensorMeta(
      DataType::INT64,
      paddle::framework::make_ddim({static_cast<int64_t>(x_dims.size())}),
      pten::DataLayout::NCHW);
  // DenseTensor d_x_dims(allocator, x_dims_meta);
  DenseTensor d_x_dims =
      pten::Empty<T, Context>(dev_ctx, std::move(x_dims_meta));

  // 1. get numbers of non zero elements
  int* nums_ptr = nums.mutable_data<int>();
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(nums_ptr, 0, sizeof(int), dev_ctx.stream()));
  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, rows);
  kernel_get_non_zero_nums<<<config.block_per_grid,
                             config.thread_per_block,
                             0,
                             dev_ctx.stream()>>>(x_data, rows, cols, nums_ptr);

  // 2. copy non_zero_num to host, copy x_dims to device
  int non_zero_num = 0;
  // const auto place = BOOST_GET_CONST(paddle::platform::CUDAPlace,
  // dev_ctx.GetPlace());
  paddle::memory::Copy(paddle::platform::CPUPlace(),
                       &non_zero_num,
                       paddle::platform::CUDAPlace(),
                       nums_ptr,
                       sizeof(int),
                       dev_ctx.stream());

  paddle::memory::Copy(paddle::platform::CUDAPlace(),
                       d_x_dims.mutable_data<int64_t>(),
                       paddle::platform::CPUPlace(),
                       x_dims.Get(),
                       x_dims.size() * sizeof(x_dims[0]),
                       dev_ctx.stream());

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
  int64_t* indices_data = indices.mutable_data<int64_t>();
  T* sparse_data = values.mutable_data<T>();

  // 3. get indexs
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(nums_ptr, 0, sizeof(int), dev_ctx.stream()));
  kernel_get_non_zero_indexs<<<config.block_per_grid,
                               config.thread_per_block,
                               config.thread_per_block.x * sizeof(int64_t),
                               dev_ctx.stream()>>>(
      x_data, rows, cols, nums_ptr, indices_data);

  // 4. sort(indexs)
  thrust::sort(thrust::cuda::par.on(dev_ctx.stream()),
               indices_data,
               indices_data + non_zero_num);

  // 5. calc indices by indexs and get values by indexs
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
      indices_data,
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

  const T* x_data = values.data<T>();
  T* out_data = out->mutable_data<T>();
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

  // const auto allocator =
  //    std::make_shared<paddle::experimental::DefaultAllocator>(
  //        dev_ctx.GetPlace());
  auto sparse_offset_meta =
      pten::DenseTensorMeta(DataType::INT64,
                            paddle::framework::make_ddim({sparse_dim}),
                            pten::DataLayout::NCHW);
  // DenseTensor d_sparse_offsets(allocator, sparse_offset_meta);
  DenseTensor d_sparse_offsets =
      pten::Empty<T, Context>(dev_ctx, std::move(sparse_offset_meta));
  // auto place = BOOST_GET_CONST(paddle::platform::CUDAPlace,
  // dev_ctx.GetPlace());
  paddle::memory::Copy(paddle::platform::CUDAPlace(),
                       d_sparse_offsets.mutable_data<int64_t>(),
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

}  // namespace pten

PT_REGISTER_KERNEL(dense_to_sparse_coo,
                   GPU,
                   ALL_LAYOUT,
                   pten::DenseToSparseCooKernel,
                   float,
                   double) {}

PT_REGISTER_KERNEL(sparse_coo_to_dense,
                   GPU,
                   ALL_LAYOUT,
                   pten::SparseCooToDenseKernel,
                   float,
                   double) {}
