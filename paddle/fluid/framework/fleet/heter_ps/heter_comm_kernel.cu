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

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/phi/common/float16.h"

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_CUDA

struct GPUCustomGradMerger {
  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__ __device__ T
  operator()(const T& a, const T& b) const {
    T out;
    out.slot = a.slot;
    out.show = a.show + b.show;
    out.clk = a.clk + b.clk;
    out.lr_g = a.lr_g + b.lr_g;
    for (int i = 0; i < MF_DIM; ++i) {
      out.mf_g[i] = a.mf_g[i] + b.mf_g[i];
    }
    return out;
  }
} gpu_merger;

template <typename T>
__global__ void fill_idx_kernel(T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

// template <typename T>
// void show_tensor(T* input, size_t len, gpuStream_t stream, std::string
// name)
// {
//  T tmp[len];  // NOLINT
//  cudaMemcpyAsync(&tmp, input, sizeof(T) * len, cudaMemcpyDeviceToHost,
//  stream);
//  cudaStreamSynchronize(stream);
//  std::cout << name;
//  for (int i = 0; i < len; ++i) {
//    std::cout << ":" << tmp[i];
//  }
//  std::cout << std::endl;
//}

template <typename T>
__global__ void calc_shard_offset_kernel(T* idx,
                                         T* left,
                                         T* right,
                                         size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len - 1) {
    if (idx[i] != idx[i + 1]) {
      right[idx[i]] = i;
      left[idx[i + 1]] = i + 1;
    }
  }
  if (i == 0) {
    left[idx[i]] = i;
  }
  if (i == (len - 1)) {
    right[idx[i]] = i;
  }
}

template <typename KeyType, typename T>
__global__ void calc_shard_index_kernel(KeyType* d_keys,
                                        size_t len,
                                        T* shard_index,
                                        int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

template <typename KeyType, typename T>
__global__ void calc_node_shard_index_kernel(KeyType* d_keys,
                                             const size_t len,
                                             T* shard_index,
                                             const int total_gpu,
                                             const int node_num) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = (d_keys[i] / total_gpu) % node_num;
  }
}

template <typename KeyType, typename T>
__global__ void fill_shard_key_kernel(KeyType* d_shard_keys,
                                      KeyType* d_keys,
                                      T* idx,
                                      size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads_kernel(KeyType* d_shard_keys,
                                        KeyType* d_keys,
                                        GradType* d_shard_grads,
                                        GradType* d_grads,
                                        T* idx,
                                        size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

template <typename ValType, typename T>
__global__ void fill_dvals_kernel(ValType* d_shard_vals,
                                  ValType* d_vals,
                                  T* idx,
                                  size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

template <typename KeyType, typename GPUAccessor>
__global__ void merge_gradients_basic_kernel(const KeyType* d_keys,
                                             const uint32_t* offset,
                                             const uint32_t* fea_num,
                                             const uint32_t* index,
                                             const char* input,
                                             char* output,
                                             int n,
                                             size_t grad_value_size,
                                             const DynamicGradMerger& merger,
                                             const GPUAccessor& gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    uint32_t start = offset[i];
    uint32_t num = fea_num[i];
    int ori_index = index[start];
    float* out = reinterpret_cast<float*>(output + i * grad_value_size);
    const float* in = reinterpret_cast<const float*>(
        input + size_t(ori_index) * grad_value_size);
    merger.update_basic(out, in, gpu_accessor);
    KeyType key = d_keys[i];
    if (key != 0) {
      for (int j = 1; j < num; ++j) {
        ori_index = index[start + j];
        in = (float*)(input + size_t(ori_index) * grad_value_size);  // NOLINT
        merger.merge_basic(out, in, gpu_accessor);
      }
    }
  }
}

template <typename KeyType, typename GPUAccessor>
__global__ void merge_gradients_embedx_kernel(const KeyType* d_keys,
                                              const uint32_t* offset,
                                              const uint32_t* fea_num,
                                              const uint32_t* index,
                                              const char* input,
                                              char* output,
                                              int n,
                                              size_t grad_dim,
                                              size_t grad_value_size,
                                              const DynamicGradMerger& merger,
                                              const GPUAccessor& gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    size_t value_idx = i / grad_dim;
    const uint32_t& start = offset[value_idx];
    const uint32_t& num = fea_num[value_idx];

    double val = 0;
    uint32_t off =
        gpu_accessor.common_push_value.EmbedxGIndex() + (i % grad_dim);
    for (uint32_t j = 0; j < num; ++j) {
      val += ((float*)(&input[size_t(index[start + j]) *  // NOLINT
                              grad_value_size]))[off];
    }
    (reinterpret_cast<float*>(&output[value_idx * grad_value_size]))[off] = val;
  }
}

__global__ void split_segments_kernel(const uint32_t* d_fea_num_info,
                                      size_t n,
                                      uint32_t* d_segments,
                                      uint32_t* d_segments_num,
                                      uint32_t segment_size) {
  const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx >= n) {
    return;
  }

  auto fea_num = d_fea_num_info[tx];
  auto seg_num = (uint32_t)((fea_num - 1) / segment_size + 1);
  d_segments[tx] = seg_num;
}

__global__ void expand_segments_kernel(const uint32_t* d_fea_num_info,
                                       const uint32_t* d_segments_offset,
                                       size_t n,
                                       uint32_t* d_segments_fea_num_info,
                                       uint32_t segment_size) {
  const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx >= n) {
    return;
  }

  auto fea_num = d_fea_num_info[tx];
  auto seg_num = (uint32_t)((fea_num - 1) / segment_size + 1);
  auto start_pos = d_segments_offset[tx];
  auto remains = fea_num;
  int cur_seg_size = 0;
  for (size_t i = 0; i < seg_num; ++i) {
    if (remains >= segment_size) {
      cur_seg_size = segment_size;
    } else {
      cur_seg_size = remains;
    }
    d_segments_fea_num_info[start_pos + i] = cur_seg_size;
    remains -= cur_seg_size;
  }
}

template <typename KeyType>
__global__ void shrink_keys_kernel(const KeyType* d_keys,
                                   const uint32_t* d_segments_offset,
                                   KeyType* d_segments_keys,
                                   size_t n) {
  const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx >= n) {
    return;
  }

  d_segments_keys[tx] = d_keys[d_segments_offset[tx]];
}

template <typename KeyType>
__global__ void unpack_merged_vals_kernel(const KeyType* d_keys,
                                          const float* d_merged_vals,
                                          const uint32_t* d_restored_idx,
                                          float* d_out,
                                          size_t val_size,
                                          const size_t n) {
  const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx >= n) {
    return;
  }

  size_t src_val_idx = 0;
  const KeyType& key = d_keys[tx];
  if (key != 0) {
    src_val_idx = d_restored_idx[tx];
  }

  uint64_t dst_offset = uint64_t(tx) * val_size;
  float* dst =
      reinterpret_cast<float*>(reinterpret_cast<char*>(d_out) + dst_offset);
  const float* src_val = reinterpret_cast<const float*>(
      reinterpret_cast<const char*>(d_merged_vals) +
      uint64_t(src_val_idx) * val_size);

  size_t n_float = val_size / sizeof(float);
  for (size_t k = 0; k < n_float; ++k) {
    dst[k] = src_val[k];
  }
}

template <typename TUnit, typename T>
__global__ void gather_keys_kernel(TUnit* d_dest_vals,
                                   const TUnit* d_src_vals,
                                   T* idx,
                                   size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_dest_vals[i] = d_src_vals[idx[i]];
  }
}
template <typename TUnit, typename T>
__global__ void scatter_keys_kernel(TUnit* d_dest_vals,
                                    const TUnit* d_src_vals,
                                    T* idx,
                                    size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_dest_vals[idx[i]] = d_src_vals[i];
  }
}

template <typename TUnit, typename T>
__global__ void gather_dvals_by_unit_kernel(TUnit* d_dest_vals,
                                            const TUnit* d_src_vals,
                                            T* idx,
                                            size_t len,
                                            const size_t val_size_unit) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    size_t pos = idx[i / val_size_unit] * val_size_unit + (i % val_size_unit);
    d_dest_vals[i] = d_src_vals[pos];
  }
}

template <typename TUnit, typename T>
__global__ void scatter_dvals_by_unit_kernel(TUnit* d_dest_vals,
                                             const TUnit* d_src_vals,
                                             T* idx,
                                             size_t len,
                                             const size_t val_size_unit) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    size_t pos = idx[i / val_size_unit] * val_size_unit + (i % val_size_unit);
    d_dest_vals[pos] = d_src_vals[i];
  }
}

// cuda implementation of  heter_comm_kernel.h
template <typename T, typename StreamType>
void HeterCommKernel::fill_idx(T* idx,
                               int64_t len,
                               const StreamType& stream,
                               const int gpu_id) {
  platform::CUDADeviceGuard guard(gpu_id);
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);
  fill_idx_kernel<<<grid_size, block_size_, 0, stream>>>(idx, c_len);
}

template <typename T, typename StreamType>
void HeterCommKernel::calc_shard_offset(T* idx,
                                        T* left,
                                        T* right,
                                        int64_t len,
                                        int total_devs,
                                        const StreamType& stream,
                                        const int gpu_id) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);
  calc_shard_offset_kernel<<<grid_size, block_size_, 0, stream>>>(
      idx, left, right, c_len);
}

template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::calc_shard_index(KeyType* d_keys,
                                       int64_t len,
                                       T* shard_index,
                                       int total_gpu,
                                       const StreamType& stream,
                                       const int gpu_id) {
  platform::CUDADeviceGuard guard(gpu_id);
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);
  calc_shard_index_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys, c_len, shard_index, total_gpu);
}
template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::calc_node_shard_index(const KeyType* d_keys,
                                            int64_t len,
                                            T* shard_index,
                                            const int& total_devs,
                                            const int& node_num,
                                            const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);
  calc_node_shard_index_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys, c_len, shard_index, total_devs, node_num);
}

template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::fill_shard_key(KeyType* d_shard_keys,
                                     KeyType* d_keys,
                                     T* idx,
                                     int64_t len,
                                     const StreamType& stream,
                                     const int gpu_id) {
  platform::CUDADeviceGuard guard(gpu_id);
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);
  fill_shard_key_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, idx, c_len);
}

template <typename KeyType, typename GradType, typename T, typename StreamType>
void HeterCommKernel::fill_shard_grads(KeyType* d_shard_keys,
                                       KeyType* d_keys,
                                       GradType* d_shard_grads,
                                       GradType* d_grads,
                                       T* idx,
                                       int64_t len,
                                       const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);
  fill_shard_grads_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, d_shard_grads, d_grads, idx, c_len);
}

template <typename ValType, typename T, typename StreamType>
void HeterCommKernel::fill_dvals(ValType* d_shard_vals,
                                 ValType* d_vals,
                                 T* idx,
                                 int64_t len,
                                 const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);
  fill_dvals_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_vals, d_vals, idx, c_len);
}

template <typename KeyT, typename ValueT, typename StreamType>
void HeterCommKernel::sort_pairs(void* d_temp_storage,
                                 size_t& temp_storage_bytes,  // NOLINT
                                 const KeyT* d_keys_in,       // NOLINT
                                 KeyT* d_keys_out,
                                 const ValueT* d_values_in,
                                 ValueT* d_values_out,
                                 int num_items,
                                 const int gpu_id,
                                 int begin_bit,
                                 int end_bit,
                                 StreamType stream,
                                 bool debug_synchronous) {
  platform::CUDADeviceGuard guard(gpu_id);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                      temp_storage_bytes,
                                      d_keys_in,
                                      d_keys_out,
                                      d_values_in,
                                      d_values_out,
                                      num_items,
                                      begin_bit,
                                      end_bit,
                                      stream,
                                      debug_synchronous));
}

template <typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename StreamType>
void HeterCommKernel::reduce_by_key(void* d_temp_storage,
                                    size_t& temp_storage_bytes,  // NOLINT
                                    KeysInputIteratorT d_keys_in,
                                    UniqueOutputIteratorT d_unique_out,
                                    ValuesInputIteratorT d_values_in,
                                    AggregatesOutputIteratorT d_aggregates_out,
                                    NumRunsOutputIteratorT d_num_runs_out,
                                    int num_items,
                                    StreamType stream,
                                    bool debug_synchronous) {
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_keys_in,
                                                            d_unique_out,
                                                            d_values_in,
                                                            d_aggregates_out,
                                                            d_num_runs_out,
                                                            gpu_merger,
                                                            num_items,
                                                            stream,
                                                            debug_synchronous));
}
template <typename KeyType,
          typename T,
          typename StreamType,
          typename GPUAccessor>
void HeterCommKernel::dy_mf_fill_shard_grads(KeyType* d_shard_keys,
                                             KeyType* d_keys,
                                             float* d_shard_grads,
                                             float* d_grads,
                                             T* idx,
                                             int64_t len,
                                             size_t grad_value_size,
                                             const StreamType& stream,
                                             const GPUAccessor& gpu_accessor) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = static_cast<size_t>(len);

  const size_t grad_value_size_float = size_t(grad_value_size / sizeof(float));
  // d_keys to d_shard_keys
  fill_shard_key_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, idx, c_len);

  PADDLE_ENFORCE_EQ(grad_value_size % sizeof(float),
                    0,
                    common::errors::InvalidArgument(
                        "The size of 'grad_value_size' must be a multiple of "
                        "sizeof(float), but received size %d.",
                        grad_value_size));

  size_t N = len * grad_value_size_float;
  grid_size = (N - 1) / block_size_ + 1;
  gather_dvals_by_unit_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_grads, d_grads, idx, N, grad_value_size_float);
}

template <typename KeyType, typename StreamType, typename GPUAccessor>
void HeterCommKernel::merge_gradient(const KeyType* d_keys,
                                     const uint32_t* offset,
                                     const uint32_t* fea_num,
                                     const uint32_t* index,
                                     const char* input,
                                     char* output,
                                     int n,
                                     size_t grad_dim,
                                     size_t grad_value_size,
                                     const DynamicGradMerger& merger,
                                     const StreamType& stream,
                                     const GPUAccessor& gpu_accessor) {
  int grid_size1 = (n - 1) / block_size_ + 1;
  merge_gradients_basic_kernel<<<grid_size1, block_size_, 0, stream>>>(
      d_keys,
      offset,
      fea_num,
      index,
      input,
      output,
      n,
      grad_value_size,
      merger,
      gpu_accessor);
  if (grad_dim > 0) {
    int grid_size2 = (n * grad_dim - 1) / block_size_ + 1;
    merge_gradients_embedx_kernel<<<grid_size2, block_size_, 0, stream>>>(
        d_keys,
        offset,
        fea_num,
        index,
        input,
        output,
        n * grad_dim,
        grad_dim,
        grad_value_size,
        merger,
        gpu_accessor);
  }
}

template <typename T, typename StreamType>
void HeterCommKernel::dy_mf_fill_dvals(float* d_shard_vals,
                                       float* d_vals,
                                       T* idx,
                                       int64_t len,
                                       size_t val_size,
                                       const StreamType& stream) {
  const size_t val_size_float = val_size / sizeof(float);
  PADDLE_ENFORCE_EQ(val_size % sizeof(float),
                    0,
                    common::errors::InvalidArgument(
                        "The size of 'val_size' must be a multiple "
                        "of sizeof(float), but received size %d.",
                        val_size));

  size_t N = len * val_size_float;
  const int grid_size = (N - 1) / block_size_ + 1;
  // fill by float, d_shard_vals to d_vals
  scatter_dvals_by_unit_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_vals, d_shard_vals, idx, N, val_size_float);
}

template <typename StreamType>
void HeterCommKernel::split_segments(const uint32_t* d_fea_num_info,
                                     size_t n,
                                     uint32_t* d_segments,
                                     uint32_t* d_segments_num,
                                     size_t segment_size,
                                     const StreamType& stream) {
  int grid_size = (n - 1) / block_size_ + 1;
  split_segments_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_fea_num_info, n, d_segments, d_segments_num, segment_size);
}

template <typename StreamType>
void HeterCommKernel::expand_segments(const uint32_t* d_fea_num_info,
                                      const uint32_t* d_segments_offset,
                                      size_t n,
                                      uint32_t* d_segments_fea_num_info,
                                      uint32_t segment_size,
                                      const StreamType& stream) {
  int grid_size = (n - 1) / block_size_ + 1;
  expand_segments_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_fea_num_info,
      d_segments_offset,
      n,
      d_segments_fea_num_info,
      segment_size);
}

template <typename KeyType, typename StreamType>
void HeterCommKernel::shrink_keys(const KeyType* d_keys,
                                  const uint32_t* d_segments_offset,
                                  KeyType* d_segments_keys,
                                  size_t n,
                                  const StreamType& stream) {
  int grid_size = (n - 1) / block_size_ + 1;
  shrink_keys_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys, d_segments_offset, d_segments_keys, n);
}
template <typename T>
__global__ void kernel_fill_restore_idx(const size_t N,
                                        const T* d_sorted_idx,
                                        const T* d_offset,
                                        const T* d_merged_cnts,
                                        T* d_restore_idx) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const T& off = d_offset[i];
    const T& num = d_merged_cnts[i];
    for (size_t k = 0; k < num; ++k) {
      d_restore_idx[d_sorted_idx[off + k]] = i;
    }
  }
}
template <typename KeyType, typename T>
__global__ void kernel_fill_restore_idx_filter_zero(const size_t N,
                                                    const KeyType* d_keys,
                                                    const T* d_sorted_idx,
                                                    const T* d_offset,
                                                    const T* d_merged_cnts,
                                                    T* d_restore_idx) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (d_keys[i] == 0) {
      return;
    }
    const T& off = d_offset[i];
    const T& num = d_merged_cnts[i];
    for (size_t k = 0; k < num; ++k) {
      d_restore_idx[d_sorted_idx[off + k]] = i;
    }
  }
}
template <typename T>
__global__ void kernel_fill_restore_idx_by_search(const size_t N,
                                                  const T* d_sorted_idx,
                                                  const size_t merge_num,
                                                  const T* d_offset,
                                                  T* d_restore_idx) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (i < d_offset[1]) {
      d_restore_idx[d_sorted_idx[i]] = 0;
      return;
    }
    int high = merge_num - 1;
    int low = 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < d_offset[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    d_restore_idx[d_sorted_idx[i]] = low;
  }
}
template <typename KeyType, typename StreamType>
void HeterCommKernel::fill_restore_idx(bool filter_zero,
                                       const size_t total_num,
                                       const size_t merge_size,
                                       const KeyType* d_keys,
                                       const uint32_t* d_sorted_idx,
                                       const uint32_t* d_offset,
                                       const uint32_t* d_merged_cnts,
                                       uint32_t* d_restore_idx,
                                       const StreamType& stream) {
  // fill restore idx [1,3,5,2,4,6] = [1,2,1,3,2,1]
  if (merge_size * 3 > total_num) {
    // repetition rate is not very high
    size_t grid_size = (merge_size - 1) / block_size_ + 1;
    if (filter_zero) {
      kernel_fill_restore_idx_filter_zero<<<grid_size,
                                            block_size_,
                                            0,
                                            stream>>>(merge_size,
                                                      d_keys,
                                                      d_sorted_idx,
                                                      d_offset,
                                                      d_merged_cnts,
                                                      d_restore_idx);
    } else {
      kernel_fill_restore_idx<<<grid_size, block_size_, 0, stream>>>(
          merge_size, d_sorted_idx, d_offset, d_merged_cnts, d_restore_idx);
    }
  } else {
    size_t grid_size = (total_num - 1) / block_size_ + 1;
    // mid search
    kernel_fill_restore_idx_by_search<<<grid_size, block_size_, 0, stream>>>(
        total_num, d_sorted_idx, merge_size, d_offset, d_restore_idx);
  }
}
template <typename KeyType, typename StreamType>
void HeterCommKernel::unpack_merged_vals(size_t n,
                                         const KeyType* d_keys,
                                         const void* d_merged_vals,
                                         const uint32_t* d_restore_idx,
                                         void* d_vals,
                                         size_t val_size,
                                         const StreamType& stream) {
  int grid_size = (n - 1) / block_size_ + 1;
  unpack_merged_vals_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys,
      reinterpret_cast<const float*>(d_merged_vals),
      d_restore_idx,
      reinterpret_cast<float*>(d_vals),
      val_size,
      n);
}
template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::gather_keys(KeyType* d_shard_keys,
                                  const KeyType* d_keys,
                                  T* idx,
                                  int64_t len,
                                  const StreamType& stream,
                                  const int gpu_id) {
  size_t N = len;
  int grid_size = (N - 1) / block_size_ + 1;
  // d_keys -> d_shard_keys
  gather_keys_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, idx, N);
}
template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::scatter_keys(const KeyType* d_shard_keys,
                                   KeyType* d_keys,
                                   T* idx,
                                   int64_t len,
                                   const StreamType& stream) {
  size_t N = len;
  int grid_size = (N - 1) / block_size_ + 1;
  // d_shard_keys -> d_keys
  scatter_keys_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys, d_shard_keys, idx, N);
}
template <typename T, typename StreamType>
void HeterCommKernel::gather_vals(float* d_shard_vals,
                                  const float* d_vals,
                                  T* idx,
                                  int64_t len,
                                  size_t value_bytes,
                                  const StreamType& stream) {
  const size_t value_size_float = size_t(value_bytes / sizeof(float));
  size_t N = len * value_size_float;
  int grid_size = (N - 1) / block_size_ + 1;
  // d_vals -> d_shard_vals
  gather_dvals_by_unit_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_vals, d_vals, idx, N, value_size_float);
}
template <typename ValType, typename StreamType>
void HeterCommKernel::scatter_vals(const ValType* d_shard_vals,
                                   ValType* d_vals,
                                   uint32_t* idx,
                                   int64_t len,
                                   size_t value_bytes,
                                   const StreamType& stream) {
  const size_t val_size_unit = size_t(value_bytes / sizeof(ValType));
  PADDLE_ENFORCE_EQ(value_bytes % sizeof(ValType),
                    0,
                    common::errors::InvalidArgument(
                        "The size of 'value_bytes' must be a multiple of "
                        "sizeof(ValType), but received size %d.",
                        value_bytes));

  size_t N = len * val_size_unit;
  const int grid_size = (N - 1) / block_size_ + 1;
  // fill by float, d_shard_vals to d_vals
  scatter_dvals_by_unit_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_vals, d_shard_vals, idx, N, val_size_unit);
}

template <typename KeyType>
__global__ void check_valid_values_kernel(const int type,
                                          const size_t N,
                                          const KeyType* keys,
                                          const char* input,
                                          const size_t value_bytes,
                                          const int num,
                                          bool debug) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const float* val = (const float*)(input + i * value_bytes);
    if (debug && (i == 0 || i == (N - 1))) {
      if (keys != nullptr) {
        printf(
            "type=%d, id=%lu, bytes=%lu, key=%lu, "
            "values=[%f,%f,%f,%f,%f,%f,%f,%f]\n",
            type,
            i,
            value_bytes,
            uint64_t(keys[i]),
            val[0],
            val[1],
            val[2],
            val[3],
            val[4],
            val[5],
            val[6],
            val[7]);
      } else {
        printf("type=%d, id=%lu, bytes=%lu, values=[%f,%f,%f,%f,%f,%f,%f,%f]\n",
               type,
               i,
               value_bytes,
               val[0],
               val[1],
               val[2],
               val[3],
               val[4],
               val[5],
               val[6],
               val[7]);
      }
    }
    for (int k = 0; k < num; ++k) {
      auto& c = val[k];
      if (isnan(c)) {
        if (keys != nullptr) {
          printf(
              "nan type %d, id=%lu, offset=%d, float=%f, key=%lu, "
              "values=[%f,%f,%f,%f,%f,%f,%f,%f]\n",
              type,
              i,
              k,
              c,
              uint64_t(keys[i]),
              val[0],
              val[1],
              val[2],
              val[3],
              val[4],
              val[5],
              val[6],
              val[7]);
        } else {
          printf("nan type %d, id=%lu, offset=%d, float=%f\n", type, i, k, c);
        }
      } else if (isinf(c)) {
        if (keys != nullptr) {
          printf(
              "inf type %d, id=%lu, offset=%d, float=%f, key=%lu, "
              "values=[%f,%f,%f,%f,%f,%f,%f,%f]\n",
              type,
              i,
              k,
              c,
              uint64_t(keys[i]),
              val[0],
              val[1],
              val[2],
              val[3],
              val[4],
              val[5],
              val[6],
              val[7]);
        } else {
          printf("inf type %d, id=%lu, offset=%d, float=%f\n", type, i, k, c);
        }
      } else if (static_cast<int>(c) > 1e+30 ||
                 static_cast<int>(c) < -(1e+30)) {
        if (keys != nullptr) {
          printf(
              "err type %d, id=%lu, offset=%d, float=%f, key=%lu, "
              "values=[%f,%f,%f,%f,%f,%f,%f,%f]\n",
              type,
              i,
              k,
              c,
              uint64_t(keys[i]),
              val[0],
              val[1],
              val[2],
              val[3],
              val[4],
              val[5],
              val[6],
              val[7]);
        } else {
          printf("err type %d, id=%lu, offset=%d, float=%f, int=%d\n",
                 type,
                 i,
                 k,
                 c,
                 static_cast<int>(c));
        }
      }
    }
  }
}
template <typename KeyType, typename StreamType>
void HeterCommKernel::check_valid_values(const int& type,
                                         const size_t& N,
                                         const KeyType* keys,
                                         const char* input,
                                         const size_t& value_bytes,
                                         const StreamType& stream,
                                         bool debug) {
  PADDLE_ENFORCE_EQ(value_bytes % sizeof(float),
                    0,
                    common::errors::InvalidArgument(
                        "The size of 'value_bytes' must be a multiple of "
                        "sizeof(float), but received size %d.",
                        value_bytes));

  const int grid_size = (N - 1) / block_size_ + 1;
  const int num = static_cast<int>(value_bytes / sizeof(float));
  check_valid_values_kernel<<<grid_size, block_size_, 0, stream>>>(
      type, N, keys, input, value_bytes, num, debug);
}

template <typename GPUAccessor>
__global__ void scale_grad_kernel(const size_t N,
                                  char* grads,
                                  const size_t value_bytes,
                                  const size_t grad_dim,
                                  const GPUAccessor& accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    size_t idx = i / grad_dim;
    size_t field_id = i % grad_dim;

    float* vals = reinterpret_cast<float*>(&grads[idx * value_bytes]);
    float& show = vals[accessor.common_push_value.ShowIndex()];
    if (show > 0.0) {
      vals[accessor.common_push_value.EmbedGIndex() + field_id] /= show;
    }
  }
}

template <typename StreamType, typename GPUAccessor>
void HeterCommKernel::scale_grad(const size_t& len,
                                 char* grads,
                                 const size_t& value_bytes,
                                 const size_t& max_mif_dim,
                                 const StreamType& stream,
                                 const GPUAccessor& gpu_accessor) {
  const size_t grad_dim = (max_mif_dim + 1);
  const size_t N = len * grad_dim;
  const int grid_size = (N - 1) / block_size_ + 1;
  scale_grad_kernel<<<grid_size, block_size_, 0, stream>>>(
      N, grads, value_bytes, grad_dim, gpu_accessor);
}
__device__ __forceinline__ int16_t float_int16(const float& val,
                                               const float& bound) {
  if (val >= bound) {
    return 32767;
  } else if (val <= -bound) {
    return -32767;
  }
  if (val > 0.0) {
    return int16_t((val * 32767.0 / bound) + 0.5);
  }
  return int16_t((val * 32767.0 / bound) - 0.5);
}
__global__ void compress_kernel(const size_t N,
                                const float* in,
                                const size_t float_num,
                                const size_t head_off,
                                char* out,
                                const size_t new_bytes,
                                const float bound) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    size_t idx = i / float_num;
    size_t off = i % float_num;

    if (off < head_off) {
      *(reinterpret_cast<float*>(&out[idx * new_bytes + off * sizeof(float)])) =
          in[i];
    } else {
      int16_t* dest = reinterpret_cast<int16_t*>(
          &out[idx * new_bytes + head_off * sizeof(float)]);
      dest[off - head_off] = float_int16(in[i], bound);
    }
  }
}
// compress
template <typename StreamType>
size_t HeterCommKernel::compress_values(const size_t& len,
                                        const char* in_vals,
                                        char* out_vals,
                                        const size_t& value_bytes,
                                        const size_t& embedx_dim,
                                        const float& max_bound,
                                        const StreamType& stream) {
  const size_t new_bytes = value_bytes - sizeof(int16_t) * embedx_dim;
  const size_t float_num = size_t(value_bytes / sizeof(float));
  const size_t head_off = float_num - embedx_dim;
  const size_t N = len * float_num;
  const int grid_size = (N - 1) / block_size_ + 1;
  compress_kernel<<<grid_size, block_size_, 0, stream>>>(N,
                                                         (const float*)in_vals,
                                                         float_num,
                                                         head_off,
                                                         out_vals,
                                                         new_bytes,
                                                         max_bound);
  return new_bytes;
}
__device__ __forceinline__ float int16_float(const int16_t& val,
                                             const float& bound) {
  return static_cast<float>(val * bound / 32767.0);
}
__global__ void uncompress_kernel(const size_t N,
                                  const char* in,
                                  const size_t new_bytes,
                                  float* out,
                                  const size_t float_num,
                                  const size_t head_off,
                                  const float bound) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    size_t idx = i / float_num;
    size_t off = i % float_num;

    if (off < head_off) {
      out[i] = *((const float*)&in[idx * new_bytes + off * sizeof(float)]);
    } else {
      const int16_t* src =
          (const int16_t*)(&in[idx * new_bytes + head_off * sizeof(float)]);
      out[i] = int16_float(src[off - head_off], bound);
    }
  }
}
// uncompress
template <typename StreamType>
void HeterCommKernel::uncompress_values(const size_t& len,
                                        const char* in_vals,
                                        char* out_vals,
                                        const size_t& value_bytes,
                                        const size_t& embedx_dim,
                                        const float& max_bound,
                                        const StreamType& stream) {
  const size_t new_bytes = value_bytes - sizeof(int16_t) * embedx_dim;
  const size_t float_num = size_t(value_bytes / sizeof(float));
  const size_t head_off = float_num - embedx_dim;
  const size_t N = len * float_num;
  const int grid_size = (N - 1) / block_size_ + 1;
  uncompress_kernel<<<grid_size, block_size_, 0, stream>>>(
      N,
      in_vals,
      new_bytes,
      reinterpret_cast<float*>(out_vals),
      float_num,
      head_off,
      max_bound);
}

template void HeterCommKernel::fill_idx<int, cudaStream_t>(
    int* idx, int64_t len, const cudaStream_t& stream, const int gpu_id);
template void HeterCommKernel::fill_idx<uint32_t, cudaStream_t>(
    uint32_t* idx, int64_t len, const cudaStream_t& stream, const int gpu_id);

template void HeterCommKernel::calc_shard_offset<int, cudaStream_t>(
    int* idx,
    int* left,
    int* right,
    int64_t len,
    int total_devs,
    const cudaStream_t& stream,
    const int gpu_id);
template void HeterCommKernel::calc_shard_offset<uint32_t, cudaStream_t>(
    uint32_t* idx,
    uint32_t* left,
    uint32_t* right,
    int64_t len,
    int total_devs,
    const cudaStream_t& stream,
    const int gpu_id);

template void HeterCommKernel::calc_shard_index<uint64_t, int, cudaStream_t>(
    uint64_t* d_keys,
    int64_t len,
    int* shard_index,
    int total_devs,
    const cudaStream_t& stream,
    const int gpu_id);

template void HeterCommKernel::calc_shard_index<int64_t, int, cudaStream_t>(
    int64_t* d_keys,
    int64_t len,
    int* shard_index,
    int total_devs,
    const cudaStream_t& stream,
    const int gpu_id);

template void
HeterCommKernel::calc_shard_index<uint64_t, uint32_t, cudaStream_t>(
    uint64_t* d_keys,
    int64_t len,
    uint32_t* shard_index,
    int total_devs,
    const cudaStream_t& stream,
    const int gpu_id);

template void
HeterCommKernel::calc_shard_index<int64_t, uint32_t, cudaStream_t>(
    int64_t* d_keys,
    int64_t len,
    uint32_t* shard_index,
    int total_devs,
    const cudaStream_t& stream,
    const int gpu_id);

template void
HeterCommKernel::calc_node_shard_index<uint64_t, int, cudaStream_t>(
    const uint64_t* d_keys,
    int64_t len,
    int* shard_index,
    const int& total_devs,
    const int& node_num,
    const cudaStream_t& stream);

template void
HeterCommKernel::calc_node_shard_index<int64_t, int, cudaStream_t>(
    const int64_t* d_keys,
    int64_t len,
    int* shard_index,
    const int& total_devs,
    const int& node_num,
    const cudaStream_t& stream);

template void
HeterCommKernel::calc_node_shard_index<uint64_t, uint32_t, cudaStream_t>(
    const uint64_t* d_keys,
    int64_t len,
    uint32_t* shard_index,
    const int& total_devs,
    const int& node_num,
    const cudaStream_t& stream);

template void
HeterCommKernel::calc_node_shard_index<int64_t, uint32_t, cudaStream_t>(
    const int64_t* d_keys,
    int64_t len,
    uint32_t* shard_index,
    const int& total_devs,
    const int& node_num,
    const cudaStream_t& stream);

template void HeterCommKernel::fill_shard_key<int64_t, int, cudaStream_t>(
    int64_t* d_shard_keys,
    int64_t* d_keys,
    int* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);

template void HeterCommKernel::fill_shard_key<uint64_t, int, cudaStream_t>(
    uint64_t* d_shard_keys,
    uint64_t* d_keys,
    int* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);

template void HeterCommKernel::fill_shard_key<int64_t, uint32_t, cudaStream_t>(
    int64_t* d_shard_keys,
    int64_t* d_keys,
    uint32_t* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);

template void HeterCommKernel::fill_shard_key<uint64_t, uint32_t, cudaStream_t>(
    uint64_t* d_shard_keys,
    uint64_t* d_keys,
    uint32_t* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);

template void
HeterCommKernel::fill_shard_grads<uint32_t, float, int, cudaStream_t>(
    uint32_t* d_shard_keys,
    uint32_t* d_keys,
    float* d_shard_grads,
    float* d_grads,
    int* idx,
    int64_t len,
    const cudaStream_t& stream);

template void
HeterCommKernel::fill_dvals<paddle::framework::FeatureValue, int, cudaStream_t>(
    paddle::framework::FeatureValue* d_shard_vals,
    paddle::framework::FeatureValue* d_vals,
    int* idx,
    int64_t len,
    const cudaStream_t& stream);

template void HeterCommKernel::fill_dvals<uint32_t, int, cudaStream_t>(
    uint32_t* d_shard_vals,
    uint32_t* d_vals,
    int* idx,
    int64_t len,
    const cudaStream_t& stream);

template void HeterCommKernel::
    sort_pairs<uint32_t, paddle::framework::FeaturePushValue, cudaStream_t>(
        void* d_temp_storage,
        size_t& temp_storage_bytes,  // NOLINT
        const uint32_t* d_keys_in,   // NOLINT
        uint32_t* d_keys_out,
        const paddle::framework::FeaturePushValue* d_values_in,
        paddle::framework::FeaturePushValue* d_values_out,
        int num_items,
        const int gpu_id,
        int begin_bit,
        int end_bit,
        cudaStream_t stream,
        bool debug_synchronous);

template void HeterCommKernel::sort_pairs<int, int, cudaStream_t>(
    void* d_temp_storage,
    size_t& temp_storage_bytes,  // NOLINT
    const int* d_keys_in,        // NOLINT
    int* d_keys_out,
    const int* d_values_in,
    int* d_values_out,
    int num_items,
    const int gpu_id,
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    bool debug_synchronous);

template void HeterCommKernel::sort_pairs<uint32_t, uint32_t, cudaStream_t>(
    void* d_temp_storage,
    size_t& temp_storage_bytes,  // NOLINT
    const uint32_t* d_keys_in,   // NOLINT
    uint32_t* d_keys_out,
    const uint32_t* d_values_in,
    uint32_t* d_values_out,
    int num_items,
    const int gpu_id,
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    bool debug_synchronous);

template void HeterCommKernel::reduce_by_key<
    uint32_t*,
    uint32_t*,
    paddle::framework::FeaturePushValue*,
    paddle::framework::FeaturePushValue*,
    int*,
    cudaStream_t>(void* d_temp_storage,
                  size_t& temp_storage_bytes,  // NOLINT
                  uint32_t* d_keys_in,
                  uint32_t* d_unique_out,
                  paddle::framework::FeaturePushValue* d_values_in,
                  paddle::framework::FeaturePushValue* d_aggregates_out,
                  int* d_num_runs_out,
                  int num_items,
                  cudaStream_t stream,
                  bool debug_synchronous);

template void HeterCommKernel::dy_mf_fill_shard_grads<
    uint64_t,
    int,
    cudaStream_t,
    CommonFeatureValueAccessor>(uint64_t* d_shard_keys,
                                uint64_t* d_keys,
                                float* d_shard_grads,
                                float* d_grads,
                                int* idx,
                                int64_t len,
                                size_t grad_value_size,
                                const cudaStream_t& stream,
                                const CommonFeatureValueAccessor& gpu_accessor);

template void HeterCommKernel::
    merge_gradient<uint32_t, cudaStream_t, CommonFeatureValueAccessor>(
        const uint32_t* d_keys,
        const uint32_t* offset,
        const uint32_t* fea_num,
        const uint32_t* index,
        const char* input,
        char* output,
        int n,
        size_t grad_dim,
        size_t grad_value_size,
        const DynamicGradMerger& merger_,
        const cudaStream_t& stream,
        const CommonFeatureValueAccessor& gpu_accessor);

template void HeterCommKernel::
    merge_gradient<uint64_t, cudaStream_t, CommonFeatureValueAccessor>(
        const uint64_t* d_keys,
        const uint32_t* offset,
        const uint32_t* fea_num,
        const uint32_t* index,
        const char* input,
        char* output,
        int n,
        size_t grad_dim,
        size_t grad_value_size,
        const DynamicGradMerger& merger_,
        const cudaStream_t& stream,
        const CommonFeatureValueAccessor& gpu_accessor);

template void HeterCommKernel::dy_mf_fill_dvals<int, cudaStream_t>(
    float* d_shard_vals,
    float* d_vals,
    int* idx,
    int64_t len,
    size_t val_size,
    const cudaStream_t& stream);

template void HeterCommKernel::dy_mf_fill_dvals<uint32_t, cudaStream_t>(
    float* d_shard_vals,
    float* d_vals,
    uint32_t* idx,
    int64_t len,
    size_t val_size,
    const cudaStream_t& stream);

template void HeterCommKernel::split_segments<cudaStream_t>(
    const uint32_t* d_fea_num_info,
    size_t n,
    uint32_t* d_segment,
    uint32_t* d_segments_num,
    size_t segment_size,
    const cudaStream_t& stream);

template void HeterCommKernel::expand_segments<cudaStream_t>(
    const uint32_t* d_fea_num_info,
    const uint32_t* d_segments_offset,
    size_t n,
    uint32_t* d_segments_fea_num_info,
    uint32_t segment_size,
    const cudaStream_t& stream);

template void HeterCommKernel::shrink_keys<uint32_t, cudaStream_t>(
    const uint32_t* d_keys,
    const uint32_t* d_segments_offset,
    uint32_t* d_segments_keys,
    size_t segment_num,
    const cudaStream_t& stream);

template void HeterCommKernel::shrink_keys<uint64_t, cudaStream_t>(
    const uint64_t* d_keys,
    const uint32_t* d_segments,
    uint64_t* d_segments_keys,
    size_t total_segment_num,
    const cudaStream_t& stream);

template void HeterCommKernel::fill_restore_idx<uint64_t, cudaStream_t>(
    bool filter_zero,
    const size_t total_num,
    const size_t merge_size,
    const uint64_t* d_keys,
    const uint32_t* d_sorted_idx,
    const uint32_t* d_offset,
    const uint32_t* d_merged_cnts,
    uint32_t* d_restore_idx,
    const cudaStream_t& stream);

template void HeterCommKernel::fill_restore_idx<uint32_t, cudaStream_t>(
    bool filter_zero,
    const size_t total_num,
    const size_t merge_size,
    const uint32_t* d_keys,
    const uint32_t* d_sorted_idx,
    const uint32_t* d_offset,
    const uint32_t* d_merged_cnts,
    uint32_t* d_restore_idx,
    const cudaStream_t& stream);

template void HeterCommKernel::unpack_merged_vals<uint64_t, cudaStream_t>(
    size_t n,
    const uint64_t* d_keys,
    const void* d_merged_vals,
    const uint32_t* d_restore_idx,
    void* d_vals,
    size_t val_size,
    const cudaStream_t& stream);

template void HeterCommKernel::unpack_merged_vals<uint32_t, cudaStream_t>(
    size_t n,
    const uint32_t* d_keys,
    const void* d_merged_vals,
    const uint32_t* d_restore_idx,
    void* d_vals,
    size_t val_size,
    const cudaStream_t& stream);

template void HeterCommKernel::gather_keys<uint64_t, int, cudaStream_t>(
    uint64_t* d_shard_keys,
    const uint64_t* d_keys,
    int* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);
template void HeterCommKernel::gather_keys<int32_t, int, cudaStream_t>(
    int32_t* d_shard_keys,
    const int32_t* d_keys,
    int* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);
template void HeterCommKernel::gather_keys<uint64_t, uint32_t, cudaStream_t>(
    uint64_t* d_shard_keys,
    const uint64_t* d_keys,
    uint32_t* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);
template void HeterCommKernel::gather_keys<int32_t, uint32_t, cudaStream_t>(
    int32_t* d_shard_keys,
    const int32_t* d_keys,
    uint32_t* idx,
    int64_t len,
    const cudaStream_t& stream,
    const int gpu_id);
template void HeterCommKernel::scatter_keys<uint64_t, int, cudaStream_t>(
    const uint64_t* d_shard_keys,
    uint64_t* d_keys,
    int* idx,
    int64_t len,
    const cudaStream_t& stream);

template void HeterCommKernel::scatter_keys<int32_t, int, cudaStream_t>(
    const int32_t* d_shard_keys,
    int32_t* d_keys,
    int* idx,
    int64_t len,
    const cudaStream_t& stream);
template void HeterCommKernel::scatter_keys<uint64_t, uint32_t, cudaStream_t>(
    const uint64_t* d_shard_keys,
    uint64_t* d_keys,
    uint32_t* idx,
    int64_t len,
    const cudaStream_t& stream);
template void HeterCommKernel::scatter_keys<int32_t, uint32_t, cudaStream_t>(
    const int32_t* d_shard_keys,
    int32_t* d_keys,
    uint32_t* idx,
    int64_t len,
    const cudaStream_t& stream);
template void HeterCommKernel::gather_vals<int, cudaStream_t>(
    float* d_shard_vals,
    const float* d_vals,
    int* idx,
    int64_t len,
    size_t value_bytes,
    const cudaStream_t& stream);
template void HeterCommKernel::gather_vals<uint32_t, cudaStream_t>(
    float* d_shard_vals,
    const float* d_vals,
    uint32_t* idx,
    int64_t len,
    size_t value_bytes,
    const cudaStream_t& stream);
template void HeterCommKernel::scatter_vals<float, cudaStream_t>(
    const float* d_shard_vals,
    float* d_vals,
    uint32_t* idx,
    int64_t len,
    size_t value_bytes,
    const cudaStream_t& stream);
template void HeterCommKernel::scatter_vals<uint64_t, cudaStream_t>(
    const uint64_t* d_shard_vals,
    uint64_t* d_vals,
    uint32_t* idx,
    int64_t len,
    size_t value_bytes,
    const cudaStream_t& stream);
template void HeterCommKernel::scatter_vals<int, cudaStream_t>(
    const int* d_shard_vals,
    int* d_vals,
    uint32_t* idx,
    int64_t len,
    size_t value_bytes,
    const cudaStream_t& stream);
template void HeterCommKernel::scatter_vals<uint32_t, cudaStream_t>(
    const uint32_t* d_shard_vals,
    uint32_t* d_vals,
    uint32_t* idx,
    int64_t len,
    size_t value_bytes,
    const cudaStream_t& stream);
template void HeterCommKernel::scatter_vals<uint8_t, cudaStream_t>(
    const uint8_t* d_shard_vals,
    uint8_t* d_vals,
    uint32_t* idx,
    int64_t len,
    size_t value_bytes,
    const cudaStream_t& stream);
template void HeterCommKernel::check_valid_values<int32_t, cudaStream_t>(
    const int& type,
    const size_t& N,
    const int32_t* keys,
    const char* input,
    const size_t& value_bytes,
    const cudaStream_t& stream,
    bool debug);
template void HeterCommKernel::check_valid_values<uint64_t, cudaStream_t>(
    const int& type,
    const size_t& N,
    const uint64_t* keys,
    const char* input,
    const size_t& value_bytes,
    const cudaStream_t& stream,
    bool debug);
template void
HeterCommKernel::scale_grad<cudaStream_t, CommonFeatureValueAccessor>(
    const size_t& len,
    char* grads,
    const size_t& value_bytes,
    const size_t& grad_dim,
    const cudaStream_t& stream,
    const CommonFeatureValueAccessor& gpu_accessor);
// compress
template size_t HeterCommKernel::compress_values<cudaStream_t>(
    const size_t& len,
    const char* in_vals,
    char* out_vals,
    const size_t& value_bytes,
    const size_t& embedx_dim,
    const float& max_bound,
    const cudaStream_t& stream);
// uncompress
template void HeterCommKernel::uncompress_values<cudaStream_t>(
    const size_t& len,
    const char* in_vals,
    char* out_vals,
    const size_t& value_bytes,
    const size_t& embedx_dim,
    const float& max_bound,
    const cudaStream_t& stream);
#endif

}  // namespace framework
}  // namespace paddle
#endif
