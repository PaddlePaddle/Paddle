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
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/gpu/utils.h"
#include "paddle/pten/kernels/sparse/cuda/sparse_coo_tensor_utils.h"

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
  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    if (!dev_is_zero(dense_data + i * cols, cols)) {
      auto index = atomicAdd(non_zero_num, 1);
      indexs[index] = i;
    }
  }
}

template <typename T>
__global__ void kernel_get_values_and_calc_indices(const T* dense_data,
                                                   const int64_t sparse_dim,
                                                   const int64_t cols,
                                                   const int64_t* src_dims,
                                                   const int non_zero_num,
                                                   const int64_t* indexs,
                                                   int64_t* indices,
                                                   T* sparse_data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int64_t sparse_index = indexs[i];
    int64_t src_index = sparse_index;
    for (int64_t j = sparse_dim - 1; j >= 0; j--) {
      indices[j * non_zero_num + i] = sparse_index % src_dims[j];
      sparse_index /= src_dims[j];
    }

    for (int j = 0; j < cols; j++) {
      sparse_data[i * cols + j] = dense_data[src_index * cols + j];
    }
  }
}

template <typename T>
void ToSparseCoo(const CUDAContext& dev_ctx,
                 const DenseTensor& src,
                 const int64_t sparse_dim,
                 SparseCooTensor* dst) {
  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();
  auto dims_2d = flatten_to_2d(src_dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          dev_ctx.GetPlace());
  auto nums_meta = pten::DenseTensorMeta(DataType::INT32,
                                         paddle::framework::make_ddim({1}),
                                         pten::DataLayout::NCHW);
  DenseTensor nums(allocator, nums_meta);
  auto src_dims_meta = pten::DenseTensorMeta(
      DataType::INT64,
      paddle::framework::make_ddim({static_cast<int64_t>(src_dims.size())}),
      pten::DataLayout::NCHW);
  DenseTensor d_src_dims(allocator, src_dims_meta);

  // 1. get numbers of non zero elements
  int* nums_ptr = nums.mutable_data<int>();
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(nums_ptr, 0, sizeof(int), dev_ctx.stream()));
  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, rows);
  kernel_get_non_zero_nums<<<config.block_per_grid, config.thread_per_block>>>(
      src_data, rows, cols, nums_ptr);

  // 2. copy nnz to host, copy src_dims to device
  int nnz = 0;
  auto place = BOOST_GET_CONST(paddle::platform::CUDAPlace, dev_ctx.GetPlace());
  paddle::memory::Copy(paddle::platform::CPUPlace(),
                       &nnz,
                       place,
                       nums_ptr,
                       sizeof(int),
                       dev_ctx.stream());
  dev_ctx.Wait();

  paddle::memory::Copy(place,
                       d_src_dims.mutable_data<int64_t>(),
                       paddle::platform::CPUPlace(),
                       src_dims.Get(),
                       src_dims.size() * sizeof(src_dims[0]),
                       dev_ctx.stream());

  dst->Resize(src_dims, sparse_dim, nnz);
  int64_t* indices_data = dst->mutable_non_zero_indices();
  T* sparse_data = dst->mutable_non_zero_elements<T>();

  // 3. get indexs
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(nums_ptr, 0, sizeof(int), dev_ctx.stream()));
  kernel_get_non_zero_indexs<<<config.block_per_grid,
                               config.thread_per_block>>>(
      src_data, rows, cols, nums_ptr, indices_data);

  // 4. sort(indexs)
  thrust::sort(
      thrust::cuda::par.on(dev_ctx.stream()), indices_data, indices_data + nnz);

  // 5. calc indices by indexs and get values by indexs
  paddle::platform::GpuLaunchConfig config2 =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, nnz);
  kernel_get_values_and_calc_indices<<<config2.block_per_grid,
                                       config2.thread_per_block>>>(
      src_data,
      sparse_dim,
      cols,
      d_src_dims.data<int64_t>(),
      nnz,
      indices_data,
      indices_data,
      sparse_data);
}

template <typename T>
void SparseCooToDense(const CUDAContext& dev_ctx,
                      const SparseCooTensor& src,
                      DenseTensor* dst) {}

}  // namespace pten

PT_REGISTER_KERNEL(
    to_sparse_coo, GPU, ALL_LAYOUT, pten::ToSparseCoo, float, double) {}
