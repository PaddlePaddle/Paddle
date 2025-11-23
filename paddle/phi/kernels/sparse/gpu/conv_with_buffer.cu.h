/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/kernels/sparse/conv_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/scatter.cu.h"
#include "paddle/phi/kernels/funcs/sparse/utils.cu.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"
#include "paddle/phi/kernels/sparse/gpu/conv_host_buffer.h"

namespace phi {
namespace sparse {
using Dims4D = phi::funcs::sparse::Dims4D;

inline __device__ uint32_t BitCount(const uint32_t data) {
  uint32_t count = data;
  count = (count & 0x55555555) + ((count >> 1) & 0x55555555);
  count = (count & 0x33333333) + ((count >> 2) & 0x33333333);
  count = (count & 0x0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f);
  count = (count & 0x00ff00ff) + ((count >> 8) & 0x00ff00ff);
  count = (count & 0x0000ffff) + ((count >> 16) & 0x0000ffff);
  return count;
}

static __global__ void GetOutIndicesCounter(const int* flags,
                                            const int n,
                                            int* out) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int block_count;
  if (threadIdx.x == 0) {
    block_count = 0;
  }
  __syncthreads();

  if (tid < n) {
    // get the count of 1 in flags[tid]
    uint32_t count = BitCount(static_cast<uint32_t>(flags[tid]));
    // add to block_count
    // TODO(zhangkaihuo): replace with block reduce_sum
    atomicAdd(&block_count, static_cast<int>(count));
  }
  __syncthreads();
  // write to out
  if (threadIdx.x == 0) {
    out[blockIdx.x] = block_count;
  }
}

// unique the out indices in rulebook
template <typename IntT>
__global__ void UniqueKernel(const IntT* in_indices,
                             int* rulebook_len,
                             int* index_flags,
                             int* out_indices,
                             int* nnz) {
  extern __shared__ int cache[];
  __shared__ int count, start;
  if (threadIdx.x == 0) {
    count = 0;
    start = 0;
  }
  __syncthreads();
  int rulebook_len_num = rulebook_len[0] / 2;

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < rulebook_len_num) {
    // atomicOr only support int
    int index = static_cast<int>((in_indices + rulebook_len_num)[i]);
    const bool flag = phi::funcs::sparse::SetBits(index, index_flags);
    if (!flag) {
      int j = atomicAdd(&count, 1);
      cache[j] = index;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    start = atomicAdd(nnz, count);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    out_indices[start + i] = cache[i];
  }
}

struct is_equal {
  int value;
  __host__ __device__ bool operator()(int x) const { return x == value; }
};

template <typename T, typename Pred>
__global__ void mark_kernel(const T* input, int* flags, Pred pred, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    flags[idx] = !pred(input[idx]);
  }
}

template <typename T>
__global__ void compact_kernel(const T* input,
                               T* output,
                               const int* indices,
                               const int* flags,
                               int n,
                               int* out_num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (flags[idx]) output[indices[idx]] = input[idx];
    if (idx == n - 1) out_num[0] = indices[idx] + flags[idx];
  }
}

template <typename T, typename Pred>
void cuda_remove(const GPUContext& dev_ctx,
                 DenseTensor& input,  // NOLINT
                 int n,
                 Pred pred,
                 int* out_num_ptr) {
  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;
  DenseTensor flags = phi::Empty<int>(dev_ctx, {n});
  DenseTensor indices = phi::Empty<int>(dev_ctx, {n});
  DenseTensor out = phi::Empty<T>(dev_ctx, {n});

  mark_kernel<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      input.data<T>(), flags.data<int>(), pred, n);

  size_t temp_size = 0;
  cub::DeviceScan::ExclusiveSum(NULL,
                                temp_size,
                                flags.data<int>(),
                                indices.data<int>(),
                                n,
                                dev_ctx.stream());
  phi::Allocator* allocator =
      const_cast<phi::Allocator*>(&(dev_ctx.GetAllocator()));
  auto ws = allocator->Allocate(temp_size)->ptr();
  cub::DeviceScan::ExclusiveSum(ws,
                                temp_size,
                                flags.data<int>(),
                                indices.data<int>(),
                                n,
                                dev_ctx.stream());

  compact_kernel<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      input.data<T>(),
      out.data<T>(),
      indices.data<int>(),
      flags.data<int>(),
      n,
      out_num_ptr);

  phi::backends::gpu::GpuMemcpyAsync(input.data<T>(),
                                     out.data<T>(),
                                     sizeof(T) * n,
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
}

template <int BS>
__global__ void GetOutIndices(const int* flags,
                              const int n,
                              const int* offsets,
                              const int out_nnz,
                              int* out) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int block_counts[BS];
  __shared__ int block_outs[BS * 32];

  int count = 0;

  if (tid < n) {
    // get the count of 1 in flags[tid]
    int flag = flags[tid];
    count = BitCount(static_cast<uint32_t>(flag));
  }

  // call block prefix_sum
  // using namespace cub;
  typedef cub::BlockScan<int, BS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  BlockScan(temp_storage).ExclusiveSum(count, count);
  __syncthreads();

  // write index to out
  if (tid < n) {
    // get the count of 1 in flags[tid]
    int flag = flags[tid];
    // int j = block_counts[threadIdx.x];
    int j = count;
    // TODO(zhangkaihuo): opt the loop
    for (int i = 0; i < 32; ++i) {
      if ((1 & (flag >> i)) == 1) {
        block_outs[j++] = (tid << 5) + i;
      }
    }
  }

  __syncthreads();
  // write to block_outs
  int start = offsets[blockIdx.x];
  int end = blockIdx.x == gridDim.x - 1 ? out_nnz : offsets[blockIdx.x + 1];
  for (int i = threadIdx.x; i < end - start; i += blockDim.x) {
    out[start + i] = block_outs[i];
  }
}

template <typename IntT>
__global__ void GroupIndices(const int* out_index_table,
                             const int n,
                             const int kernel_size,
                             IntT* out_indices,
                             int* out_index_counts,
                             int* out_index_groups) {
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    IntT index = out_indices[i];
    int real_index = out_index_table[index];
    out_indices[i] = real_index;

    // kernel_size at most
    int j = atomicAdd(out_index_counts + real_index, 1);
    // nnz * kernel_size
    out_index_groups[real_index * kernel_size + j] = i;
  }
}

template <typename IntT>
__global__ void GetOutIndexTable(int* indices,
                                 const int non_zero_num,
                                 const Dims4D out_dims,
                                 const bool is2D,
                                 int* out_index_table,
                                 IntT* out_indices) {
  CUDA_KERNEL_LOOP_TYPE(i, non_zero_num, int64_t) {
    IntT index = static_cast<IntT>(indices[i]);
    out_index_table[index] = i;
    IntT batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<Dims4D>(
        index, out_dims, &batch, &x, &y, &z);
    // get out indices
    out_indices[i] = batch;
    if (is2D) {
      out_indices[i + non_zero_num] = y;
      out_indices[i + non_zero_num * 2] = x;
    } else {
      out_indices[i + non_zero_num] = z;
      out_indices[i + non_zero_num * 2] = y;
      out_indices[i + non_zero_num * 3] = x;
    }
    indices[i] = 0;
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
 * the calculation
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
                                      const bool is2D,
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
    T in_z = is2D ? 0 : x_indices[i + non_zero_num];
    T in_y =
        is2D ? x_indices[i + non_zero_num] : x_indices[i + 2 * non_zero_num];
    T in_x = is2D ? x_indices[i + 2 * non_zero_num]
                  : x_indices[i + 3 * non_zero_num];
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
            T out_z =
                is2D ? 0
                     : (in_z + paddings[1] - kz * dilations[1]) / strides[1];
            T out_y = (in_y + paddings[2] - ky * dilations[2]) / strides[2];
            T out_x = (in_x + paddings[3] - kx * dilations[3]) / strides[3];
            in_i = i;
            out_index = phi::funcs::sparse::PointToIndex<Dims4D>(
                batch, out_x, out_y, out_z, out_dims);
            atomicAdd(&counter_buf[kernel_index], 1);
            kernel_i = kernel_index;
          }
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

template <typename T, typename Context, typename IntT = int>
int ProductRuleBookWithBuffer(const Context& dev_ctx,
                              const IntT* indices_ptr,
                              const Dims4D& d_x_dims,
                              const Dims4D& d_kernel_dims,
                              const Dims4D& d_out_dims,
                              const Dims4D& d_paddings,
                              const Dims4D& d_strides,
                              const Dims4D& d_dilations,
                              const DDim& out_dims,
                              const std::vector<int>& kernel_sizes,
                              const int64_t& non_zero_num,
                              const int& kernel_size,
                              const int& rulebook_rows,
                              const int& rulebook_cols,
                              IntT* rulebook_ptr,
                              int* counter_ptr,
                              int* offsets_ptr,
                              DenseTensor* index_flags,
                              DenseTensor* out_index_table,
                              DenseTensor* rulebook,
                              DenseTensor* out_index,
                              DenseTensor* unique_value,
                              SparseCooTensor* out,
                              int* h_buffer) {
  DenseTensor d_buffer = phi::Empty<int>(dev_ctx, {2 * kernel_size + 3});
  const bool is2D = out_dims.size() == 4 ? true : false;
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);
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
                                                    is2D,
                                                    rulebook_ptr,
                                                    counter_ptr);

  DenseTensor rulebook_len_tensor = phi::Empty<int>(dev_ctx, {1});
  cuda_remove<IntT>(dev_ctx,
                    *rulebook,
                    rulebook_rows * rulebook_cols,
                    is_equal{-1},
                    rulebook_len_tensor.data<int>());

  size_t temp_size = 0;
  cub::DeviceScan::ExclusiveSum(
      NULL, temp_size, counter_ptr, offsets_ptr, kernel_size, dev_ctx.stream());
  phi::Allocator* allocator =
      const_cast<phi::Allocator*>(&(dev_ctx.GetAllocator()));
  auto ws = allocator->Allocate(temp_size)->ptr();
  cub::DeviceScan::ExclusiveSum(
      ws, temp_size, counter_ptr, offsets_ptr, kernel_size, dev_ctx.stream());

  int64_t max_nnz =
      phi::sparse::ConvHostBuffer::getInstance().get_max_bound() * non_zero_num;
  rulebook->Resize({rulebook_rows, static_cast<int>(max_nnz)});
  // 3. sorted or merge the out index

  out_index->ResizeAndAllocate({static_cast<int>(max_nnz)});
  DenseTensor unique_key =
      phi::Empty<int>(dev_ctx, {static_cast<int>(max_nnz)});
  int* out_index_ptr = out_index->data<int>();
  int* unique_key_ptr = unique_key.data<int>();

  phi::backends::gpu::GpuMemsetAsync(
      unique_key_ptr, 0, sizeof(int), dev_ctx.stream());
  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, max_nnz, 1);
  size_t cache_size = sizeof(int) * config.thread_per_block.x;
  int* index_flags_ptr = index_flags->data<int>();
  UniqueKernel<IntT><<<config.block_per_grid,
                       config.thread_per_block,
                       cache_size,
                       dev_ctx.stream()>>>(rulebook_ptr,
                                           rulebook_len_tensor.data<int>(),
                                           index_flags_ptr,
                                           out_index_ptr,
                                           unique_key_ptr);

  phi::backends::gpu::GpuMemcpyAsync(d_buffer.data<int>(),
                                     counter_ptr,
                                     kernel_size * sizeof(int),
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(d_buffer.data<int>() + kernel_size,
                                     offsets_ptr,
                                     kernel_size * sizeof(int),
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(d_buffer.data<int>() + 2 * kernel_size + 1,
                                     rulebook_len_tensor.data<int>(),
                                     sizeof(int),
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(d_buffer.data<int>() + 2 * kernel_size + 2,
                                     unique_key_ptr,
                                     sizeof(int),
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(h_buffer,
                                     d_buffer.data<int>(),
                                     (2 * kernel_size + 3) * sizeof(int),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());

  dev_ctx.Wait();
  int rulebook_len = h_buffer[2 * kernel_size + 1] / 2;
  int out_nnz = h_buffer[2 * kernel_size + 2];

  rulebook->Resize({rulebook_rows, static_cast<int>(rulebook_len)});
  out_index->Resize({static_cast<int>(rulebook_len)});

  const int threads = 256;
  const int blocks = (index_flags->numel() + threads - 1) / threads;
  int* out_index_table_ptr = out_index_table->data<int>();

  GetOutIndicesCounter<<<blocks, threads, 0, dev_ctx.stream()>>>(
      index_flags_ptr, index_flags->numel(), out_index_table_ptr);

  size_t temp_size1 = 0;
  cub::DeviceScan::ExclusiveSum(NULL,
                                temp_size1,
                                out_index_table_ptr,
                                out_index_table_ptr,
                                blocks,
                                dev_ctx.stream());

  phi::Allocator* allocator1 =
      const_cast<phi::Allocator*>(&(dev_ctx.GetAllocator()));
  auto ws1 = allocator->Allocate(temp_size)->ptr();
  cub::DeviceScan::ExclusiveSum(ws1,
                                temp_size1,
                                out_index_table_ptr,
                                out_index_table_ptr,
                                blocks,
                                dev_ctx.stream());

  GetOutIndices<threads>
      <<<blocks, threads, 0, dev_ctx.stream()>>>(index_flags_ptr,
                                                 index_flags->numel(),
                                                 out_index_table_ptr,
                                                 out_nnz,
                                                 out_index_ptr);

  const int64_t sparse_dim = is2D ? 3 : 4;
  phi::DenseTensor out_indices =
      phi::Empty<IntT>(dev_ctx, {sparse_dim, out_nnz});

  phi::DenseTensor out_values =
      phi::Empty<T>(dev_ctx, {out_nnz, kernel_sizes[sparse_dim]});
  out->SetMember(out_indices, out_values, out_dims, false);

  IntT* out_indices_ptr = out_indices.data<IntT>();

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_nnz, 1);
  GetOutIndexTable<IntT>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          out_index_ptr,
          out_nnz,
          d_out_dims,
          is2D,
          out_index_table_ptr,
          out_indices_ptr);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
  unique_value->ResizeAndAllocate({static_cast<int>(out_nnz * kernel_size)});
  int* unique_value_ptr = unique_value->data<int>();

  GroupIndices<<<config.block_per_grid,
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
}  // namespace sparse
}  // namespace phi
