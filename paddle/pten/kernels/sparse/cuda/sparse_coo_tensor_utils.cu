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

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/sparse.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/pten/api/lib/utils/allocator.h"
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
                                         const int64_t batch_size,
                                         const int rows,
                                         const int cols,
                                         int64_t* non_zero_nums) {
  int64_t batch = blockIdx.y;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int64_t counter;
  if (threadIdx.x == 0) counter = 0;
  __syncthreads();

  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    if (!is_zero(dense_data + i * cols, cols)) {
      atomicAdd(&counter, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(non_zero_nums + batch, counter);
  }
}

template <typename T>
__global__ void kernel_get_non_zero_indexs(const T* dense_data,
                                           const int64_t batch_size,
                                           const int rows,
                                           const int cols,
                                           int64_t* non_zero_nums,
                                           int64_t* indexs) {
  int64_t batch = blockIdx.y;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    if (!is_zero(dense_data + i * cols, cols)) {
      int index = atomicAdd(&non_zero_nums[batch], 1);
      indexs[index] = i;
    }
  }
}

// Currently, this kernel only supports batch_size = 1
template <typename T>
__global__ void kernel_get_values_and_calc_indices(const T* dense_data,
                                                   const int64_t batch_size,
                                                   const int64_t sparse_dim,
                                                   const int rows,
                                                   const int cols,
                                                   const int64_t* src_dims,
                                                   const int64_t* non_zero_nums,
                                                   const int64_t* indexs,
                                                   int64_t* indices,
                                                   T* sparse_data) {
  int64_t batch = blockIdx.y;
  const int non_zero_num = non_zero_nums[batch];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    int64_t sparse_index = indexs[i];
    int64_t src_index = sparse_index;
    for (int64_t j = sparse_dim - 1; j >= 0; j--) {
      indices_data[j * non_zero_num + i] = sparse_index % src_dims[j];
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

  // get non zero numbers
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          dev_ctx.GetPlace());
  const int batch_size = 1;
  auto nums_meta = pten::DenseTensorMeta(DataType::INT64,
                                         framework::make_ddim({batch_size}),
                                         pten::DataLayout::NCHW);
  DenseTensor nums(allocator, nums_meta);
  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, rows);
  config.block_per_grid.y = batch_size;
  kenrel_get_non_zero_nums<<<config.block_per_grid, config.thread_per_block>>>(
      src_data, batch_size, rows, cols, nums.data<int64_t>());

  // alloc indexs space
  std::vector<int64_t> h_nums(batch_size);
  auto place = BOOST_GET_CONST(paddle::platform::CUDAPlace, dev_ctx.GetPlace());
  paddle::memory::Copy(paddle::platform::CPUPlace(),
                       h_nums.data(),
                       place,
                       nums.data<int64_t>(),
                       sizeof(int64_t) * batch_size);
  int64_t total_nnz = 0;
  for (int64_t i = 0; i < batch_size; i++) {
    total_nnz += h_nums[i];
  }

  dst->Resize(src_dims, sparse_dim, total_nnz);
  int64_t* indices_data = dst->mutable_non_zero_indices();
  T* values_data = dst->mutable_non_zero_elements<T>();

  // get indexs and values
  kernel_get_non_zero_indexs<<<config.block_per_grid,
                               config.thread_per_block>>>(src_data,
                                                          batch_size,
                                                          rows,
                                                          cols,
                                                          nums.data<int64_t>(),
                                                          indices_data,
                                                          values_data);

  // sort(indexs)
  thrust::sort(
      thrust::pair::cuda(dev_ctx.stream()), indices, indices + total_nnz);

  // calc indices by indexs and get values by indexs
  // kernel_get_values_and_calc_indices<<<config.block_per_grid,
  // config.thread_per_block>>>(src_data, batch_size, rows, cols,
}

template <typename T>
void SparseCooToDense(const CUDAContext& dev_ctx,
                      const SparseCooTensor& src,
                      DenseTensor* dst) {}

}  // namespace pten
