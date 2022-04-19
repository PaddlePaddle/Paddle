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
__global__ void calc_shard_offset_kernel(T* idx, T* left, T* right,
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
__global__ void calc_shard_index_kernel(KeyType* d_keys, size_t len,
                                        T* shard_index, int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

template <typename KeyType, typename T>
__global__ void fill_shard_key_kernel(KeyType* d_shard_keys, KeyType* d_keys,
                                      T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads_kernel(KeyType* d_shard_keys, KeyType* d_keys,
                                        GradType* d_shard_grads,
                                        GradType* d_grads, T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

template <typename ValType, typename T>
__global__ void fill_dvals_kernel(ValType* d_shard_vals, ValType* d_vals,
                                  T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

// cuda implemention of  heter_comm_kernel.h
template <typename T, typename StreamType>
void HeterCommKernel::fill_idx(T* idx, long long len,
                               const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  fill_idx_kernel<<<grid_size, block_size_, 0, stream>>>(idx, c_len);
}

template <typename T, typename StreamType>
void HeterCommKernel::calc_shard_offset(T* idx, T* left, T* right,
                                        long long len, int total_devs,
                                        const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  calc_shard_offset_kernel<<<grid_size, block_size_, 0, stream>>>(idx, left,
                                                                  right, c_len);
}

template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::calc_shard_index(KeyType* d_keys, long long len,
                                       T* shard_index, int total_gpu,
                                       const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  calc_shard_index_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys, c_len, shard_index, total_gpu);
}

template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys,
                                     T* idx, long long len,
                                     const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  fill_shard_key_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, idx, c_len);
}

template <typename KeyType, typename GradType, typename T, typename StreamType>
void HeterCommKernel::fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                       GradType* d_shard_grads,
                                       GradType* d_grads, T* idx, long long len,
                                       const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  fill_shard_grads_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, d_shard_grads, d_grads, idx, c_len);
}

template <typename ValType, typename T, typename StreamType>
void HeterCommKernel::fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                                 long long len, const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  fill_dvals_kernel<<<grid_size, block_size_, 0, stream>>>(d_shard_vals, d_vals,
                                                           idx, c_len);
}

template <typename KeyT, typename ValueT, typename StreamType>
void HeterCommKernel::sort_pairs(void* d_temp_storage,
                                 size_t& temp_storage_bytes,  // NOLINT
                                 const KeyT* d_keys_in,       // NOLINT
                                 KeyT* d_keys_out, const ValueT* d_values_in,
                                 ValueT* d_values_out, int num_items,
                                 int begin_bit, int end_bit, StreamType stream,
                                 bool debug_synchronous) {
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, begin_bit, end_bit, stream, debug_synchronous));
}

template <typename KeysInputIteratorT, typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT, typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT, typename StreamType>
void HeterCommKernel::reduce_by_key(void* d_temp_storage,
                                    size_t& temp_storage_bytes,  // NOLINT
                                    KeysInputIteratorT d_keys_in,
                                    UniqueOutputIteratorT d_unique_out,
                                    ValuesInputIteratorT d_values_in,
                                    AggregatesOutputIteratorT d_aggregates_out,
                                    NumRunsOutputIteratorT d_num_runs_out,
                                    int num_items, StreamType stream,
                                    bool debug_synchronous) {
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_unique_out, d_values_in,
      d_aggregates_out, d_num_runs_out, gpu_merger, num_items, stream,
      debug_synchronous));
}

template void HeterCommKernel::fill_idx<int, cudaStream_t>(
    int* idx, long long len, const cudaStream_t& stream);

template void HeterCommKernel::calc_shard_offset<int, cudaStream_t>(
    int* idx, int* left, int* right, long long len, int total_devs,
    const cudaStream_t& stream);
template void HeterCommKernel::calc_shard_index<
    unsigned long, int, cudaStream_t>(unsigned long* d_keys, long long len,
                                      int* shard_index, int total_devs,
                                      const cudaStream_t& stream);

template void HeterCommKernel::calc_shard_index<long, int, cudaStream_t>(
    long* d_keys, long long len, int* shard_index, int total_devs,
    const cudaStream_t& stream);

template void HeterCommKernel::fill_shard_key<long, int, cudaStream_t>(
    long* d_shard_keys, long* d_keys, int* idx, long long len,
    const cudaStream_t& stream);

template void HeterCommKernel::fill_shard_key<unsigned long, int, cudaStream_t>(
    unsigned long* d_shard_keys, unsigned long* d_keys, int* idx, long long len,
    const cudaStream_t& stream);

template void HeterCommKernel::fill_shard_grads<
    unsigned long, paddle::framework::FeaturePushValue, int, cudaStream_t>(
    unsigned long* d_shard_keys, unsigned long* d_keys,
    paddle::framework::FeaturePushValue* d_shard_grads,
    paddle::framework::FeaturePushValue* d_grads, int* idx, long long len,
    const cudaStream_t& stream);

template void
HeterCommKernel::fill_dvals<paddle::framework::FeatureValue, int, cudaStream_t>(
    paddle::framework::FeatureValue* d_shard_vals,
    paddle::framework::FeatureValue* d_vals, int* idx, long long len,
    const cudaStream_t& stream);

template void HeterCommKernel::sort_pairs<
    unsigned long, paddle::framework::FeaturePushValue, cudaStream_t>(
    void* d_temp_storage,
    size_t& temp_storage_bytes,      // NOLINT
    const unsigned long* d_keys_in,  // NOLINT
    unsigned long* d_keys_out,
    const paddle::framework::FeaturePushValue* d_values_in,
    paddle::framework::FeaturePushValue* d_values_out, int num_items,
    int begin_bit, int end_bit, cudaStream_t stream, bool debug_synchronous);

template void HeterCommKernel::sort_pairs<int, int, cudaStream_t>(
    void* d_temp_storage,
    size_t& temp_storage_bytes,  // NOLINT
    const int* d_keys_in,        // NOLINT
    int* d_keys_out, const int* d_values_in, int* d_values_out, int num_items,
    int begin_bit, int end_bit, cudaStream_t stream, bool debug_synchronous);

template void HeterCommKernel::reduce_by_key<
    unsigned long*, unsigned long*, paddle::framework::FeaturePushValue*,
    paddle::framework::FeaturePushValue*, int*, cudaStream_t>(
    void* d_temp_storage,
    size_t& temp_storage_bytes,  // NOLINT
    unsigned long* d_keys_in, unsigned long* d_unique_out,
    paddle::framework::FeaturePushValue* d_values_in,
    paddle::framework::FeaturePushValue* d_aggregates_out, int* d_num_runs_out,
    int num_items, cudaStream_t stream, bool debug_synchronous);

#endif

}  // namespace framework
}  // namespace paddle
#endif
