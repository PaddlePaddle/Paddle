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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/sparse/common_shape.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
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
      phi::DenseTensorMeta(DataType::INT32, {1}, phi::DataLayout::NCHW);
  DenseTensor nums = phi::Empty(dev_ctx, std::move(nums_meta));
  auto x_dims_meta = phi::DenseTensorMeta(DataType::INT64,
                                          {static_cast<int64_t>(x_dims.size())},
                                          phi::DataLayout::NCHW);
  DenseTensor d_x_dims = phi::Empty(dev_ctx, std::move(x_dims_meta));

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
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rows, 1);

  auto temp_indexs_meta =
      phi::DenseTensorMeta(DataType::INT32, {rows}, phi::DataLayout::NCHW);
  DenseTensor temp_indexs = phi::Empty(dev_ctx, std::move(temp_indexs_meta));
  int* temp_indexs_ptr = temp_indexs.mutable_data<int>(place);
  GetNonZeroNums<<<config.block_per_grid.x,
                   config.thread_per_block.x,
                   0,
                   dev_ctx.stream()>>>(
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

  const auto values_dims =
      phi::funcs::sparse::InferDenseDims(x_dims, sparse_dim, non_zero_num);
  DenseTensorMeta indices_meta(DataType::INT64,
                               {sparse_dim, static_cast<int64_t>(non_zero_num)},
                               DataLayout::NCHW);
  DenseTensorMeta values_meta(x.meta().dtype, values_dims, x.meta().layout);
  phi::DenseTensor indices(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(indices_meta));
  phi::DenseTensor values(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(values_meta));
  int64_t* indices_data = indices.mutable_data<int64_t>(place);
  T* sparse_data = values.mutable_data<T>(place);

  // 3. calc indices by indexs and get values by indexs
  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);
  GetNonZeroElementsAndIndices<<<config.block_per_grid.x,
                                 config.thread_per_block.x,
                                 0,
                                 dev_ctx.stream()>>>(x_data,
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
  DenseTensorMeta values_meta(
      x.dtype(), {non_zero_num}, x.non_zero_elements().layout());
  DenseTensorMeta offsets_meta(DataType::INT32, {batchs}, DataLayout::NCHW);
  DenseTensor indices = phi::Empty(dev_ctx, std::move(indices_meta));
  DenseTensor values = phi::Empty(dev_ctx, std::move(values_meta));
  DenseTensor offsets = phi::Empty(dev_ctx, std::move(offsets_meta));
  int64_t* coo_indices = indices.mutable_data<int64_t>(place);
  int64_t* batch_ptr = x_dims.size() == 2 ? nullptr : coo_indices;
  int64_t* coo_rows_data =
      x_dims.size() == 2 ? coo_indices : batch_ptr + non_zero_num;
  int64_t* coo_cols_data = coo_rows_data + non_zero_num;
  int* offsets_ptr = batchs == 1 ? nullptr : offsets.mutable_data<int>(place);
  T* coo_values_data = values.mutable_data<T>(place);

  if (batchs > 1) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, batchs, 1);
    GetBatchSizes<<<config.block_per_grid.x, config.thread_per_block.x>>>(
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

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rows, 1);
  config.block_per_grid.y = batchs;
  ConvertCsrCrowsToCooRows<<<config.block_per_grid,
                             config.thread_per_block.x>>>(
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

__global__ void GetBatchsOffset(const int64_t* batchs_ptr,
                                const int non_zero_num,
                                int64_t* batchs_offset) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    if (i == non_zero_num - 1 || batchs_ptr[i] != batchs_ptr[i + 1]) {
      batchs_offset[batchs_ptr[i]] = i + 1;
    }
  }
}

__global__ void ConvertCooRowsToCsrCrows(
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
                    phi::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D or 3-D matrix"));
  const int64_t non_zero_num = x.nnz();
  if (non_zero_num <= 0) return;

  int batchs = x_dims.size() == 2 ? 1 : x_dims[0];
  int rows = x_dims.size() == 2 ? x_dims[0] : x_dims[1];

  const auto place = dev_ctx.GetPlace();
  DenseTensorMeta crows_meta(
      DataType::INT64, {batchs * (rows + 1)}, DataLayout::NCHW);
  DenseTensorMeta cols_meta(DataType::INT64, {non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {non_zero_num}, x.non_zero_elements().layout());
  phi::DenseTensor non_zero_crows(
      phi::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(crows_meta));
  phi::DenseTensor non_zero_cols(
      phi::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(cols_meta));
  phi::DenseTensor non_zero_elements(
      phi::make_intrusive<paddle::experimental::SharedStorage>(place),
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

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, batchs, 1);
  if (batchs > 1) {
    DenseTensorMeta batchs_meta(DataType::INT64, {batchs}, DataLayout::NCHW);
    phi::DenseTensor batchs_offset(
        phi::make_intrusive<paddle::experimental::SharedStorage>(place),
        std::move(batchs_meta));
    int64_t* batchs_offset_ptr = batchs_offset.mutable_data<int64_t>(place);
    GetBatchsOffset<<<config.block_per_grid.x,
                      config.thread_per_block.x,
                      0,
                      dev_ctx.stream()>>>(
        batchs_ptr, non_zero_num, batchs_offset_ptr);
    config.block_per_grid.y = batchs;
    ConvertCooRowsToCsrCrows<<<config.block_per_grid,
                               config.thread_per_block.x,
                               0,
                               dev_ctx.stream()>>>(
        batchs_offset_ptr, coo_rows_data, csr_crows_data, rows, non_zero_num);
  } else {
    ConvertCooRowsToCsrCrows<<<config.block_per_grid.x,
                               config.thread_per_block.x,
                               0,
                               dev_ctx.stream()>>>(
        nullptr, coo_rows_data, csr_crows_data, rows, non_zero_num);
  }

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(csr_cols_data,
                                            coo_cols_data,
                                            sizeof(int64_t) * non_zero_num,
                                            hipMemcpyDeviceToDevice,
                                            dev_ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(csr_values_data,
                                            coo_values_data,
                                            sizeof(T) * non_zero_num,
                                            hipMemcpyDeviceToDevice,
                                            dev_ctx.stream()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(csr_cols_data,
                                             coo_cols_data,
                                             sizeof(int64_t) * non_zero_num,
                                             cudaMemcpyDeviceToDevice,
                                             dev_ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(csr_values_data,
                                             coo_values_data,
                                             sizeof(T) * non_zero_num,
                                             cudaMemcpyDeviceToDevice,
                                             dev_ctx.stream()));
#endif
  out->SetMember(non_zero_crows, non_zero_cols, non_zero_elements, x_dims);
}

template <typename ValueT, typename IndicesT>
__global__ void KernelSparseCooToDense(const IndicesT* indices,
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
  *out = phi::Empty(dev_ctx,
                    phi::DenseTensorMeta(
                        x.dtype(), x.dims(), x.non_zero_elements().layout()));
  T* out_data = out->data<T>();
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

  auto sparse_offset_meta = phi::DenseTensorMeta(
      DataType::INT64, {sparse_dim}, phi::DataLayout::NCHW);
  DenseTensor d_sparse_offsets = Empty(dev_ctx, std::move(sparse_offset_meta));

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemcpyAsync(d_sparse_offsets.mutable_data<int64_t>(place),
                     sparse_offsets.data(),
                     sparse_dim * sizeof(int64_t),
                     hipMemcpyHostToDevice,
                     dev_ctx.stream()));

  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemsetAsync(out_data, 0, sizeof(T) * out->numel(), dev_ctx.stream()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(d_sparse_offsets.mutable_data<int64_t>(place),
                      sparse_offsets.data(),
                      sparse_dim * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      dev_ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(out_data, 0, sizeof(T) * out->numel(), dev_ctx.stream()));
#endif
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);

  KernelSparseCooToDense<T, int64_t><<<config.block_per_grid.x,
                                       config.thread_per_block.x,
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

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(dense_to_sparse_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToSparseCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_csr_to_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrToCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_coo_to_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooToCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(dense_to_sparse_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToSparseCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_coo_to_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooToDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_csr_to_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrToDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(coo_values,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CooValuesKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(csr_values,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrValuesKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_coo_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooTensorKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {}
