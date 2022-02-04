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

#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/sparse/sparse_utils_kernel.h"

namespace pten {
namespace sparse {

template <typename T>
inline __device__ bool DevIsZero(const T* data, const int64_t cols) {
  const T zero = static_cast<T>(0);
  // TODO(zhangkaihuo): check the data is zero or not in parallen when cols > 1
  for (int64_t i = 0; i < cols; i++) {
    if (data[i] != zero) {
      return false;
    }
  }
  return true;
}

template <typename T>
__global__ void GetNonZeroNums(const T* dense_data,
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
    // TODO(zhangkaihuo): when cols=1, vectorization can be used
    if (!DevIsZero(dense_data + i * cols, cols)) {
      // use reductions?
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
__global__ void GetNonZeroElementsAndIndices(const T* dense_data,
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

template <typename Context>
void GetGpuLaunchConfig1D(const Context& dev_ctx,
                          const int64_t n,
                          int* grid_size,
                          int* block_size) {
  const int MAX_BLOCK_DIM = dev_ctx.GetMaxThreadsPerBlock();
  const int MAX_GRID_DIM = dev_ctx.GetMaxPhysicalThreadCount() / MAX_BLOCK_DIM;
  *block_size = (n >= MAX_BLOCK_DIM) ? MAX_BLOCK_DIM
                                     : (1 << static_cast<int>(std::log2(n)));
  *grid_size = n / *block_size;
  *grid_size = (*grid_size >= MAX_GRID_DIM) ? MAX_GRID_DIM : *grid_size;
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
  auto nums_meta =
      pten::DenseTensorMeta(DataType::INT32, {1}, pten::DataLayout::NCHW);
  DenseTensor nums = pten::Empty(dev_ctx, std::move(nums_meta));
  auto x_dims_meta =
      pten::DenseTensorMeta(DataType::INT64,
                            {static_cast<int64_t>(x_dims.size())},
                            pten::DataLayout::NCHW);
  DenseTensor d_x_dims = pten::Empty(dev_ctx, std::move(x_dims_meta));

  const auto place = dev_ctx.GetPlace();

  // 1. get numbers of non zero elements, and get the index of non zero elements
  int* nums_ptr = nums.mutable_data<int>(place);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemsetAsync(nums_ptr, 0, sizeof(int), dev_ctx.stream()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(nums_ptr, 0, sizeof(int), dev_ctx.stream()));
#endif
  int grid_size = 1, block_size = 1;
  GetGpuLaunchConfig1D(dev_ctx, rows, &grid_size, &block_size);

  auto temp_indexs_meta =
      pten::DenseTensorMeta(DataType::INT32, {rows}, pten::DataLayout::NCHW);
  DenseTensor temp_indexs = pten::Empty(dev_ctx, std::move(temp_indexs_meta));
  int* temp_indexs_ptr = temp_indexs.mutable_data<int>(place);
  GetNonZeroNums<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      x_data, rows, cols, nums_ptr, temp_indexs_ptr);
#ifdef PADDLE_WITH_HIP
  thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                 temp_indexs_ptr,
                 temp_indexs_ptr + rows,
                 -1);

  // 2. copy non_zero_num to host, copy x_dims to device
  int non_zero_num = 0;
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(&non_zero_num,
                                            nums_ptr,
                                            sizeof(int),
                                            hipMemcpyDeviceToHost,
                                            dev_ctx.stream()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&non_zero_num,
                                             nums_ptr,
                                             sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             dev_ctx.stream()));
#endif

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemcpyAsync(d_x_dims.mutable_data<int64_t>(place),
                     x_dims.Get(),
                     x_dims.size() * sizeof(x_dims[0]),
                     hipMemcpyHostToDevice,
                     dev_ctx.stream()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(d_x_dims.mutable_data<int64_t>(place),
                      x_dims.Get(),
                      x_dims.size() * sizeof(x_dims[0]),
                      cudaMemcpyHostToDevice,
                      dev_ctx.stream()));
#endif

  dev_ctx.Wait();  // wait the copy

  const auto values_dims = InferDenseDims(x_dims, sparse_dim, non_zero_num);
  DenseTensorMeta indices_meta(DataType::INT64,
                               {sparse_dim, static_cast<int64_t>(non_zero_num)},
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
  GetGpuLaunchConfig1D(dev_ctx, non_zero_num, &grid_size, &block_size);
  GetNonZeroElementsAndIndices<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
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

__global__ void GetBatchSizes(const int64_t* crows,
                              const int rows,
                              const int batchs,
                              int* batch_sizes) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < batchs) {
    batch_sizes[tid] = crows[tid * (rows + 1) + rows];
  }
}

__global__ void ConvertCsrCrowsToCooRows(const int64_t* crows_ptr,
                                         const int* crows_offsets,
                                         int64_t* rows_ptr,
                                         int64_t* batch_ptr,
                                         const int rows) {
  const int b = blockIdx.y;
  const int64_t offset = crows_offsets ? crows_offsets[b] : 0;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    for (int j = crows_ptr[b * (rows + 1) + i];
         j < crows_ptr[b * (rows + 1) + i + 1];
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
  DenseTensor indices = pten::Empty(dev_ctx, std::move(indices_meta));
  DenseTensor values = pten::Empty(dev_ctx, std::move(values_meta));
  DenseTensor offsets = pten::Empty(dev_ctx, std::move(offsets_meta));
  int64_t* coo_indices = indices.mutable_data<int64_t>(place);
  int64_t* batch_ptr = x_dims.size() == 2 ? nullptr : coo_indices;
  int64_t* coo_rows_data =
      x_dims.size() == 2 ? coo_indices : batch_ptr + non_zero_num;
  int64_t* coo_cols_data = coo_rows_data + non_zero_num;
  int* offsets_ptr = batchs == 1 ? nullptr : offsets.mutable_data<int>(place);
  T* coo_values_data = values.mutable_data<T>(place);

  int grid_size = 1, block_size = 1;
  if (batchs > 1) {
    GetGpuLaunchConfig1D(dev_ctx, batchs, &grid_size, &block_size);
    GetBatchSizes<<<grid_size, block_size>>>(
        csr_crows_data, rows, batchs, offsets_ptr);

#ifdef PADDLE_WITH_HIP
    thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
    thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                           offsets_ptr,
                           offsets_ptr + batchs,
                           offsets_ptr);
  }

  GetGpuLaunchConfig1D(dev_ctx, rows, &grid_size, &block_size);
  dim3 grids(grid_size, batchs, 1);
  ConvertCsrCrowsToCooRows<<<grids, block_size>>>(
      csr_crows_data, offsets_ptr, coo_rows_data, batch_ptr, rows);

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(coo_cols_data,
                                            csr_cols_data,
                                            sizeof(int64_t) * non_zero_num,
                                            hipMemcpyDeviceToDevice,
                                            dev_ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(coo_values_data,
                                            csr_values_data,
                                            sizeof(T) * non_zero_num,
                                            hipMemcpyDeviceToDevice,
                                            dev_ctx.stream()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(coo_cols_data,
                                             csr_cols_data,
                                             sizeof(int64_t) * non_zero_num,
                                             cudaMemcpyDeviceToDevice,
                                             dev_ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(coo_values_data,
                                             csr_values_data,
                                             sizeof(T) * non_zero_num,
                                             cudaMemcpyDeviceToDevice,
                                             dev_ctx.stream()));
#endif

  out->SetMember(indices, values, x_dims, true);
}

}  // namespace sparse
}  // namespace pten

PT_REGISTER_KERNEL(dense_to_sparse_coo,
                   GPU,
                   ALL_LAYOUT,
                   pten::sparse::DenseToSparseCooKernel,
                   float,
                   double,
                   pten::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL(sparse_csr_to_coo,
                   GPU,
                   ALL_LAYOUT,
                   pten::sparse::SparseCsrToCooKernel,
                   float,
                   double,
                   pten::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
