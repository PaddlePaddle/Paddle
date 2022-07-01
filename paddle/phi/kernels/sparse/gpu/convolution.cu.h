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

#pragma once

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/utils.cu.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"
#include "paddle/phi/kernels/sparse/convolution_kernel.h"

namespace phi {
namespace sparse {

using Dims4D = phi::funcs::sparse::Dims4D;

// TODO(zhangkaihuo): After the GatherCUDAKernel is migrated to phi, replace
// this kernel with phi::GatherCUDAKernel;
// Vectorization can be used to improve read and write bandwidth
/**
 * brief: gather data from params according to indices
 * params: the inputs
 * indices: the indices you want to gather
 * output: the outputs
 * index_size: the size of indices
 * slice_size: slice size corresponding to each index, here is the channel size
 **/
template <typename T, typename IndexT = int, int VecSize>
__global__ void GatherKernel(const T* params,
                             const IndexT* indices,
                             T* output,
                             size_t index_size,
                             size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size / VecSize, int64_t) {
    const int vec_slice_size = slice_size / VecSize;
    int indices_i = i / vec_slice_size;
    int slice_i = i - indices_i * vec_slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    int64_t params_i = gather_i * slice_size + slice_i * VecSize;
    using LoadT = phi::AlignedVector<T, VecSize>;
    using StoreT = phi::AlignedVector<T, VecSize>;
    LoadT params_vec;
    phi::Load<T, VecSize>(params + params_i, &params_vec);
    phi::Store<T, VecSize>(params_vec, output + i * VecSize);
  }
}

template <typename T, typename IntT, int VecSize>
__global__ void GatherKernelV2(const T* inputs,
                               const int* index_counts,
                               const int* origin_indexs,
                               const int non_zero_num,
                               const int kernel_size,
                               T* output,
                               const int channels) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int vec_channels = channels / VecSize;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  for (int i = tid; i < non_zero_num * vec_channels;
       i += gridDim.x * blockDim.x) {
    int indices_i = i / vec_channels;
    int channels_i = i - indices_i * vec_channels;
    int len = index_counts[indices_i];
    LoadT in_vec;
    phi::Load<T, VecSize>(inputs + indices_i * channels + channels_i * VecSize,
                          &in_vec);
    for (int j = 0; j < len; j++) {
      int out_i = origin_indexs[indices_i * kernel_size + j];
      phi::Store<T, VecSize>(in_vec,
                             output + out_i * channels + channels_i * VecSize);
    }
  }
}

template <typename T, typename IntT, int VecSize>
__global__ void GatherKernelV3(const T* inputs,
                               const int* index_counts,
                               const int* origin_indexs,
                               const int non_zero_num,
                               const int kernel_size,
                               T* output,
                               const int channels) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int vec_channels = channels / VecSize;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  for (int i = tid; i < non_zero_num * vec_channels;
       i += gridDim.x * blockDim.x) {
    int indices_i = i / vec_channels;
    int channels_i = i - indices_i * vec_channels;
    int len1 = index_counts[indices_i];
    LoadT in_vec;
    phi::Load<T, VecSize>(inputs + indices_i * channels + channels_i * VecSize,
                          &in_vec);
    for (int j = 0; j < len1; j++) {
      int out_i = origin_indexs[indices_i * kernel_size + j];
      phi::Store<T, VecSize>(in_vec,
                             output + out_i * channels + channels_i * VecSize);
    }
    int len2 = index_counts[non_zero_num + indices_i];
    for (int j = 0; j < len2; j++) {
      int out_i = origin_indexs[indices_i * kernel_size + j +
                                kernel_size * non_zero_num];
      phi::Store<T, VecSize>(in_vec,
                             output + out_i * channels + channels_i * VecSize);
    }
  }
}

template <typename IntT>
__global__ void UniqueKernel(const IntT* in_indexs,
                             const int rulebook_len,
                             int* out_index_table,
                             int* out_indexs,
                             int* nnz) {
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < rulebook_len) {
    // atomicOr only support int
    int index = static_cast<int>(in_indexs[i]);
    int change_index = index == 0 ? -1 : index;
    int flag = atomicOr(out_index_table + index, change_index);
    if (flag == 0) {
      int j = atomicAdd(&count, 1);
      cache[j] = index;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(nnz, count);
  }
}

template <typename IntT>
__global__ void UpdateOutIndex(const int* out_index_table,
                               const int n,
                               const int kernel_size,
                               IntT* out_indexs,
                               int* out_index_counts,
                               int* origin_out_indexs) {
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    IntT index = out_indexs[i];
    int real_index = out_index_table[index];
    out_indexs[i] = real_index;

    // kernel_size at most
    int j = atomicAdd(out_index_counts + real_index, 1);
    // nnz * kernel_size
    origin_out_indexs[real_index * kernel_size + j] = i;
  }
}

/**
 * @brief: update the out index and indices
 * unique_keys: save the index of the output feature list
 * unique_values: indiates the index of key before deduplication
 * out_indexs: indicates the position of the output index in the rulebook
 * rulebook_len: indicates the length of rulebook
 * out_dims: indicates the output dims
 * out_indices: the indices of output, out_indices = IndexToPoint(unique_keys)
 * rulebook_out_indexs: the output index in rulebook
 **/
template <typename T>
__global__ void UpdateIndexKernel(const T* unique_keys,
                                  const int* unique_values,
                                  const int* out_indexs,
                                  const int64_t non_zero_num,
                                  const int rulebook_len,
                                  const Dims4D out_dims,
                                  T* out_indices,
                                  T* rulebook_out_indexs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    const T index = unique_keys[i];
    T batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<Dims4D>(
        index, out_dims, &batch, &x, &y, &z);
    // get out indices
    out_indices[i] = batch;
    out_indices[i + non_zero_num] = z;
    out_indices[i + non_zero_num * 2] = y;
    out_indices[i + non_zero_num * 3] = x;

    // update rulebook
    int start = unique_values[i];
    int end = i == non_zero_num - 1 ? rulebook_len : unique_values[i + 1];
    // max(end-start) = kernel_size
    for (T j = start; j < end; j++) {
      rulebook_out_indexs[out_indexs[j]] = i;
    }
  }
}

/**
 * @brief product rulebook
 * for input_i in x_indices:
 *   if input_i participate in the convolution calculation:
 *       infer the output_i by input_i and kernel_i
 *       save output_i
 *
 * x_indices: the indices of input features
 * x_dims: the input dims
 * kernel_dims: the kernel dims
 * out_dims: the output dims
 * non_zero_num: the number of input features
 * rulebook: the rulebook to save the kernel index, input index and output index
 * counter: save the number of times each location in the kernel participates in
 *the caculation
 **/
template <typename T>
__global__ void ProductRuleBookKernel(const T* x_indices,
                                      const Dims4D x_dims,
                                      const Dims4D kernel_dims,
                                      const Dims4D out_dims,
                                      const int64_t non_zero_num,
                                      const Dims4D paddings,
                                      const Dims4D dilations,
                                      const Dims4D strides,
                                      T* rulebook,
                                      int* counter) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ int counter_buf[];  // kernel_size
  const int kernel_size = kernel_dims[3] * kernel_dims[2] * kernel_dims[1];
  const int offset = kernel_size * non_zero_num;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int kernel_index = 0;
    T batch = x_indices[i];
    T in_z = x_indices[i + non_zero_num];
    T in_y = x_indices[i + 2 * non_zero_num];
    T in_x = x_indices[i + 3 * non_zero_num];
    for (int kz = 0; kz < kernel_dims[1]; kz++) {
      for (int ky = 0; ky < kernel_dims[2]; ky++) {
        for (int kx = 0; kx < kernel_dims[3]; kx++) {
          int in_i = -1, out_index = -1, kernel_i = -1;
          if (phi::funcs::sparse::Check(x_dims,
                                        kernel_dims,
                                        paddings,
                                        dilations,
                                        strides,
                                        in_x,
                                        in_y,
                                        in_z,
                                        kx,
                                        ky,
                                        kz)) {
            T out_z = (in_z + paddings[1] - kz * dilations[1]) / strides[1];
            T out_y = (in_y + paddings[2] - ky * dilations[2]) / strides[2];
            T out_x = (in_x + paddings[3] - kx * dilations[3]) / strides[3];
            in_i = i;
            out_index = phi::funcs::sparse::PointToIndex<Dims4D>(
                batch, out_x, out_y, out_z, out_dims);
            atomicAdd(&counter_buf[kernel_index], 1);
            kernel_i = kernel_index;
          }
          // rulebook[kernel_index * non_zero_num + i] = kernel_i;
          rulebook[kernel_index * non_zero_num + i] = in_i;
          rulebook[kernel_index * non_zero_num + offset + i] = out_index;
          ++kernel_index;
        }
      }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    atomicAdd(&counter[i], counter_buf[i]);
  }
}

template <typename IntT>
__global__ void GetOutIndexTable(const IntT* indices,
                                 const IntT non_zero_num,
                                 const Dims4D dims,
                                 IntT* out_index_table) {
  CUDA_KERNEL_LOOP_TYPE(i, non_zero_num, int64_t) {
    IntT batch = indices[i];
    IntT in_z = indices[i + non_zero_num];
    IntT in_y = indices[i + 2 * non_zero_num];
    IntT in_x = indices[i + 3 * non_zero_num];
    IntT index = PointToIndex(batch, in_x, in_y, in_z, dims);
    out_index_table[index] = i;
  }
}

template <typename IntT>
__global__ void GetOutIndexTable(const int* indexs,
                                 const int non_zero_num,
                                 const Dims4D out_dims,
                                 int* out_index_table,
                                 IntT* out_indices) {
  CUDA_KERNEL_LOOP_TYPE(i, non_zero_num, int64_t) {
    IntT index = static_cast<IntT>(indexs[i]);
    index = index == -1 ? 0 : index;
    out_index_table[index] = i;
    IntT batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<Dims4D>(
        index, out_dims, &batch, &x, &y, &z);
    // get out indices
    out_indices[i] = batch;
    out_indices[i + non_zero_num] = z;
    out_indices[i + non_zero_num * 2] = y;
    out_indices[i + non_zero_num * 3] = x;
  }
}

template <typename IntT>
__global__ void CopyRuleBook(const int* counters,
                             const int* offsets,
                             const IntT* in_rulebook,
                             const int len,
                             const int kernel_size,
                             const int non_zero_num,
                             IntT* out_rulebook) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ int cache_counters[];
  int* cache_offsets = cache_counters + kernel_size;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    cache_counters[i] = counters[i];
    cache_offsets[i] = offsets[i];
  }
  __syncthreads();
  for (int i = tid; i < len; i += gridDim.x * blockDim.x) {
    // get the kernel index
    int kernel_index = 0;
    for (; kernel_index < kernel_size - 1; kernel_index++) {
      if (i >= offsets[kernel_index] && i < offsets[kernel_index + 1]) {
        break;
      }
    }
    int inner_index = i - offsets[kernel_index];
    // out_rulebook[i] = in_rulebook[kernel_index * non_zero_num + inner_index];
    out_rulebook[i] = in_rulebook[kernel_index * non_zero_num + inner_index];
    out_rulebook[len + i] =
        in_rulebook[kernel_size * non_zero_num + kernel_index * non_zero_num +
                    inner_index];
  }
}

template <typename T>
__global__ void ProductSubmRuleBookKernel(const T* x_indices,
                                          const Dims4D x_dims,
                                          const Dims4D kernel_dims,
                                          const Dims4D out_dims,
                                          const int64_t non_zero_num,
                                          const Dims4D paddings,
                                          const Dims4D dilations,
                                          const Dims4D strides,
                                          const T* out_index_table,
                                          T* rulebook,
                                          int* counter) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int kernel_size = kernel_dims[3] * kernel_dims[2] * kernel_dims[1];
  extern __shared__ int counter_buf[];  // kernel_size
  int* counter_buf2 = counter_buf + kernel_size;
  // length = kernel_size * blockDim.x * 2;
  int* rulebook_buf = counter_buf + kernel_size * 2;

  const int offset = kernel_size * non_zero_num;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int kernel_index = 0;
    T batch = x_indices[i];
    T in_z = x_indices[i + non_zero_num];
    T in_y = x_indices[i + 2 * non_zero_num];
    T in_x = x_indices[i + 3 * non_zero_num];
    for (int kz = 0; kz < kernel_dims[1]; kz++) {
      for (int ky = 0; ky < kernel_dims[2]; ky++) {
        for (int kx = 0; kx < kernel_dims[3]; kx++) {
          int in_i = -1, out_index = -1, kernel_i = -1;
          if (phi::funcs::sparse::Check(x_dims,
                                        kernel_dims,
                                        paddings,
                                        dilations,
                                        strides,
                                        in_x,
                                        in_y,
                                        in_z,
                                        kx,
                                        ky,
                                        kz)) {
            T out_z = (in_z + paddings[1] - kz * dilations[1]) / strides[1];
            T out_y = (in_y + paddings[2] - ky * dilations[2]) / strides[2];
            T out_x = (in_x + paddings[3] - kx * dilations[3]) / strides[3];
            out_index = phi::funcs::sparse::PointToIndex<Dims4D>(
                batch, out_x, out_y, out_z, out_dims);
            int real_out_index = out_index_table[out_index];
            if (real_out_index != -1) {
              in_i = i;
              int buf_i = atomicAdd(&counter_buf[kernel_index], 1);
              kernel_i = kernel_index;
              rulebook_buf[kernel_index * blockDim.x + buf_i] = in_i;
              rulebook_buf[kernel_index * blockDim.x +
                           kernel_size * blockDim.x + buf_i] = real_out_index;
            }
          }
          ++kernel_index;
        }
      }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf2[i] = atomicAdd(&counter[i], counter_buf[i]);
  }
  __syncthreads();
  for (int i = 0; i < kernel_size; i++) {
    if (threadIdx.x < counter_buf[i]) {
      // rulebook[i * non_zero_num + counter_buf2[i] + threadIdx.x] = i;
      rulebook[i * non_zero_num + counter_buf2[i] + threadIdx.x] =
          rulebook_buf[i * blockDim.x + threadIdx.x];
      rulebook[i * non_zero_num + offset + counter_buf2[i] + threadIdx.x] =
          rulebook_buf[i * blockDim.x + kernel_size * blockDim.x + threadIdx.x];
    }
  }
}

template <typename IntT>
__global__ void UpdateOutIndex(const int n,
                               const int kernel_size,
                               const IntT* indexs,
                               int* index_counts,
                               int* index_groups) {
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    IntT index = indexs[i];
    // kernel_size at most
    int j = atomicAdd(index_counts + index, 1);
    // nnz * kernel_size
    index_groups[index * kernel_size + j] = i;
  }
}

template <typename IntT>
__global__ void UpdateOutIndexV2(const int rulebook_len,
                                 const int non_zero_num,
                                 const int kernel_size,
                                 const int half_kernel_offset,
                                 const IntT* indexs,
                                 int* index_counts,
                                 int* index_groups) {
  CUDA_KERNEL_LOOP_TYPE(i, rulebook_len, int64_t) {
    IntT index = indexs[i];
    int* counts_ptr =
        i < half_kernel_offset ? index_counts : index_counts + non_zero_num;
    int* groups_ptr = i < half_kernel_offset
                          ? index_groups
                          : index_groups + non_zero_num * kernel_size;
    // conflict kernel_size times at most
    int j = atomicAdd(counts_ptr + index, 1);
    // nnz * kernel_size
    groups_ptr[index * kernel_size + j] = i;
  }
}

// the basic algorithm can refer to convolution_kernel.cc or
// the second paper
// example:
// 1. the rulebook:
//  the kernel_index:                       0, 0, 0, 1, 1, 1, 2, 2, ....
//  the out_index(key):                     20, 30, 33, 30, 33, 20, 25
// 2. mark the index of out_index(value):   0, 1, 2, 3, 4, 5, 6, ....
// 3. sorted the (key, value)
// 4. unique the (key, value):
//  unique_key:     20, 25, 30, 33
//  unique_values:  0, 2, 3, 5
//  the index of unique_values is: 0, 1, 2, 3
// 5. update the out_index by unique_key, uniqe_value and the index of
// unique_value:
//  the new out_index: 0, 2, 3, 2, 3, 0, 1
template <typename T, typename Context, typename IntT = int>
int ProductRuleBook(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const std::vector<int>& kernel_sizes,
                    const std::vector<int>& paddings,
                    const std::vector<int>& dilations,
                    const std::vector<int>& strides,
                    const DDim& out_dims,
                    const bool subm,
                    DenseTensor* rulebook,
                    DenseTensor* counter_per_kernel,
                    DenseTensor* offsets_per_kernel,
                    DenseTensor* out_index,
                    DenseTensor* unique_value,
                    SparseCooTensor* out,
                    std::vector<int>* h_counter,
                    std::vector<int>* h_offsets) {
  auto indices_dtype = paddle::experimental::CppTypeToDataType<IntT>::Type();
  const int64_t non_zero_num = x.nnz();
  const auto& non_zero_indices = x.non_zero_indices();
  const IntT* indices_ptr = non_zero_indices.data<IntT>();
  int* counter_ptr = counter_per_kernel->data<int>();
  int* offsets_ptr = offsets_per_kernel->data<int>();
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];

  const auto x_dims = x.dims();
  Dims4D d_x_dims(x_dims[0], x_dims[3], x_dims[2], x_dims[1]);
  Dims4D d_kernel_dims(1, kernel_sizes[2], kernel_sizes[1], kernel_sizes[0]);
  Dims4D d_out_dims(out_dims[0], out_dims[3], out_dims[2], out_dims[1]);
  Dims4D d_paddings(1, paddings[2], paddings[1], paddings[0]);
  Dims4D d_strides(1, strides[2], strides[1], strides[0]);
  Dims4D d_dilations(1, dilations[2], dilations[1], dilations[0]);
  // 1. product rule book
  // phi::funcs::SetConstant<Context, int> set_zero;
  // set_zero(dev_ctx, counter_per_kernel, 0);
  phi::backends::gpu::GpuMemsetAsync(counter_ptr,
                                     0,
                                     sizeof(int) * counter_per_kernel->numel(),
                                     dev_ctx.stream());
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);

  if (subm) {
    // At present, hashtable is not used to map the input and output indexes.
    // At present, the intermediate output index is generated by normal
    // convolution,
    // and then the intermediate output index is subtracted from the input index
    // to obain the rulebook.
    const int rulebook_rows = 2;
    const int rulebook_cols = kernel_size * non_zero_num;
    DenseTensorMeta rulebook_meta(
        indices_dtype, {rulebook_rows, rulebook_cols}, DataLayout::NCHW);
    DenseTensor tmp_rulebook = phi::Empty(dev_ctx, std::move(rulebook_meta));
    IntT* rulebook_ptr = tmp_rulebook.data<IntT>();
    DenseTensor out_indices =
        phi::EmptyLike<IntT>(dev_ctx, x.non_zero_indices());
    DenseTensor out_values =
        phi::Empty(dev_ctx,
                   DenseTensorMeta(x.dtype(),
                                   {x.nnz(), kernel_sizes[4]},
                                   x.non_zero_elements().layout()));
    phi::Copy(
        dev_ctx, x.non_zero_indices(), dev_ctx.GetPlace(), false, &out_indices);

    int64_t table_size = 1;
    for (int i = 0; i < out_dims.size() - 1; i++) {
      table_size *= out_dims[i];
    }
    DenseTensor out_index_table = phi::Empty<IntT>(dev_ctx, {table_size});
    IntT* out_index_table_ptr = out_index_table.data<IntT>();
    thrust::fill(thrust::cuda::par.on(dev_ctx.stream()),
                 out_index_table_ptr,
                 out_index_table_ptr + out_index_table.numel(),
                 -1);

    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);
    GetOutIndexTable<IntT><<<config.block_per_grid,
                             config.thread_per_block,
                             0,
                             dev_ctx.stream()>>>(
        out_indices.data<IntT>(), non_zero_num, d_x_dims, out_index_table_ptr);

    size_t cache_size = kernel_size * 2 + kernel_size *
                                              config.thread_per_block.x * 2 *
                                              sizeof(int);
    const int MAX_CACHE_SIZE = 48 * 1024;
    while (cache_size >= MAX_CACHE_SIZE) {
      config.thread_per_block.x /= 2;
      config.block_per_grid.x *= 2;
      PADDLE_ENFORCE_GE(config.thread_per_block.x,
                        32,
                        phi::errors::Fatal("the shared memory is not enough"));
      cache_size = kernel_size * 2 +
                   kernel_size * config.thread_per_block.x * 2 * sizeof(int);
    }
    ProductSubmRuleBookKernel<IntT><<<config.block_per_grid.x,
                                      config.thread_per_block.x,
                                      cache_size,
                                      dev_ctx.stream()>>>(indices_ptr,
                                                          d_x_dims,
                                                          d_kernel_dims,
                                                          d_out_dims,
                                                          non_zero_num,
                                                          d_paddings,
                                                          d_dilations,
                                                          d_strides,
                                                          out_index_table_ptr,
                                                          rulebook_ptr,
                                                          counter_ptr);

    out->SetMember(out_indices, out_values, out_dims, true);

    thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
                           counter_ptr,
                           counter_ptr + kernel_size,
                           offsets_ptr);

    phi::backends::gpu::GpuMemcpyAsync(&(*h_counter)[0],
                                       counter_ptr,
                                       kernel_size * sizeof(int),
                                       gpuMemcpyDeviceToHost,
                                       dev_ctx.stream());

    phi::backends::gpu::GpuMemcpyAsync(&(*h_offsets)[0],
                                       offsets_ptr,
                                       kernel_size * sizeof(int),
                                       gpuMemcpyDeviceToHost,
                                       dev_ctx.stream());
    dev_ctx.Wait();
    int rulebook_len =
        (*h_offsets)[kernel_size - 1] + (*h_counter)[kernel_size - 1];
    DenseTensor out_rulebook =
        phi::Empty<IntT>(dev_ctx, {rulebook_rows, rulebook_len});
    IntT* out_rulebook_ptr = out_rulebook.data<IntT>();
    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
    cache_size = kernel_size * 2 * sizeof(int);
    CopyRuleBook<IntT><<<config.block_per_grid,
                         config.thread_per_block,
                         cache_size,
                         dev_ctx.stream()>>>(counter_ptr,
                                             offsets_ptr,
                                             rulebook_ptr,
                                             rulebook_len,
                                             kernel_size,
                                             non_zero_num,
                                             out_rulebook_ptr);
    *rulebook = out_rulebook;

    return rulebook_len;

  } else {
    const int rulebook_rows = 2;
    const int rulebook_cols = kernel_size * non_zero_num;
    DenseTensorMeta rulebook_meta(
        indices_dtype, {rulebook_rows, rulebook_cols}, DataLayout::NCHW);
    *rulebook = phi::Empty(dev_ctx, std::move(rulebook_meta));
    IntT* rulebook_ptr = rulebook->data<IntT>();
    ProductRuleBookKernel<IntT><<<config.block_per_grid.x,
                                  config.thread_per_block.x,
                                  kernel_size * sizeof(int),
                                  dev_ctx.stream()>>>(indices_ptr,
                                                      d_x_dims,
                                                      d_kernel_dims,
                                                      d_out_dims,
                                                      non_zero_num,
                                                      d_paddings,
                                                      d_dilations,
                                                      d_strides,
                                                      rulebook_ptr,
                                                      counter_ptr);

    // 2. remove -1
#ifdef PADDLE_WITH_HIP
    IntT* last = thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
    IntT* last = thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                                rulebook_ptr,
                                rulebook_ptr + rulebook_rows * rulebook_cols,
                                -1);

    // phi::funcs::sparse::DistanceKernel<IntT><<<1, 1, 0, dev_ctx.stream()>>>(
    //     rulebook_ptr, last, rulebook_ptr + rulebook_rows * rulebook_cols -
    //     1);
    // IntT rulebook_len = 0;
    // phi::backends::gpu::GpuMemcpyAsync(
    //     &rulebook_len,
    //     rulebook_ptr + rulebook_rows * rulebook_cols - 1,
    //     sizeof(IntT),
    //     gpuMemcpyDeviceToHost,
    //     dev_ctx.stream());

    // dev_ctx.Wait();
    // rulebook_len /= 2;
    // printf("rulebook_len = %d\n", rulebook_len);
    // printf("distance = %d\n", last-rulebook_ptr);
    IntT rulebook_len = (last - rulebook_ptr) / 2;

#ifdef PADDLE_WITH_HIP
    thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
    thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                           counter_ptr,
                           counter_ptr + kernel_size,
                           offsets_ptr);

    phi::backends::gpu::GpuMemcpyAsync(&(*h_counter)[0],
                                       counter_ptr,
                                       kernel_size * sizeof(int),
                                       gpuMemcpyDeviceToHost,
                                       dev_ctx.stream());

    phi::backends::gpu::GpuMemcpyAsync(&(*h_offsets)[0],
                                       offsets_ptr,
                                       kernel_size * sizeof(int),
                                       gpuMemcpyDeviceToHost,
                                       dev_ctx.stream());

    rulebook->Resize({rulebook_rows, static_cast<int>(rulebook_len)});
    // 3. sorted or merge the out index
    out_index->ResizeAndAllocate({static_cast<int>(rulebook_len)});
    DenseTensor unique_key =
        phi::Empty<int>(dev_ctx, {static_cast<int>(rulebook_len)});
    int* out_index_ptr = out_index->data<int>();
    int* unique_key_ptr = unique_key.data<int>();

    int64_t table_size = 1;
    for (int i = 0; i < out_dims.size() - 1; i++) {
      table_size *= out_dims[i];
    }
    DenseTensor out_index_table = phi::Empty<int>(dev_ctx, {table_size});
    int* out_index_table_ptr = out_index_table.data<int>();
    cudaMemsetAsync(
        out_index_table_ptr, 0, sizeof(int) * table_size, dev_ctx.stream());
    cudaMemsetAsync(unique_key_ptr, 0, sizeof(int), dev_ctx.stream());

    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
    UniqueKernel<IntT><<<config.block_per_grid,
                         config.thread_per_block,
                         0,
                         dev_ctx.stream()>>>(rulebook_ptr + rulebook_len,
                                             rulebook_len,
                                             out_index_table_ptr,
                                             out_index_ptr,
                                             unique_key_ptr);
    int out_nnz = 0;
    cudaMemcpyAsync(&out_nnz,
                    unique_key_ptr,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    dev_ctx.stream());
    dev_ctx.Wait();

    const int64_t sparse_dim = 4;
    DenseTensorMeta indices_meta(
        indices_dtype, {sparse_dim, out_nnz}, DataLayout::NCHW);
    DenseTensorMeta values_meta(
        x.dtype(), {out_nnz, kernel_sizes[4]}, x.non_zero_elements().layout());
    phi::DenseTensor out_indices = phi::Empty(dev_ctx, std::move(indices_meta));
    phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));
    out->SetMember(out_indices, out_values, out_dims, true);

    IntT* out_indices_ptr = out_indices.data<IntT>();

    // thrust::sort(thrust::cuda::par.on(dev_ctx.stream()),
    //              out_index_ptr,
    //              out_index_ptr + out_nnz);
    thrust::remove_copy(thrust::cuda::par.on(dev_ctx.stream()),
                        out_index_table_ptr,
                        out_index_table_ptr + table_size,
                        out_index_ptr,
                        0);

    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_nnz, 1);
    GetOutIndexTable<IntT><<<config.block_per_grid,
                             config.thread_per_block,
                             0,
                             dev_ctx.stream()>>>(out_index_ptr,
                                                 out_nnz,
                                                 d_out_dims,
                                                 out_index_table_ptr,
                                                 out_indices_ptr);
    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
    cudaMemsetAsync(
        out_index_ptr, 0, sizeof(int) * rulebook_len, dev_ctx.stream());
    unique_value->ResizeAndAllocate({static_cast<int>(out_nnz * kernel_size)});
    int* unique_value_ptr = unique_value->data<int>();

    // return rulebook_len;
    UpdateOutIndex<<<config.block_per_grid,
                     config.thread_per_block,
                     0,
                     dev_ctx.stream()>>>(out_index_table_ptr,
                                         rulebook_len,
                                         kernel_size,
                                         rulebook_ptr + rulebook_len,
                                         out_index_ptr,
                                         unique_value_ptr);

    return rulebook_len;
  }
}

}  // namespace sparse
}  // namespace phi
