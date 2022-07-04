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

template <typename KeyType, typename T>
__global__ void dy_mf_fill_shard_grads_kernel(
    KeyType* d_shard_keys, KeyType* d_keys, float* d_shard_grads,
    float* d_grads, T* idx, size_t len, size_t grad_value_size,
    CommonFeatureValueAccessor feature_value_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    float* cur = (float*)((char*)d_shard_grads + i * grad_value_size);
    float* shard_val = (float*)((char*)d_grads + uint64_t(idx[i]) * grad_value_size);

    cur[feature_value_accessor.common_push_value.SlotIndex()] =
      shard_val[feature_value_accessor.common_push_value.SlotIndex()];
    cur[feature_value_accessor.common_push_value.ShowIndex()] =
      shard_val[feature_value_accessor.common_push_value.ShowIndex()];
    cur[feature_value_accessor.common_push_value.ClickIndex()] =
      shard_val[feature_value_accessor.common_push_value.ClickIndex()];
    cur[feature_value_accessor.common_push_value.MfDimIndex()] =
      shard_val[feature_value_accessor.common_push_value.MfDimIndex()];
    cur[feature_value_accessor.common_push_value.EmbedGIndex()] =
      shard_val[feature_value_accessor.common_push_value.EmbedGIndex()];

    for (int x = 0; x < int(shard_val[feature_value_accessor.common_push_value.MfDimIndex()]); x++) {
      cur[feature_value_accessor.common_push_value.EmbedxGIndex() + x] = 
        shard_val[feature_value_accessor.common_push_value.EmbedxGIndex() + x];
    }
  }
}

template <typename KeyType>
__global__ void merge_gradients_basic_kernel(const KeyType* d_keys,
                                       const uint32_t* offset,
                                       const uint32_t* fea_num,
                                       const uint32_t* index, const char* input,
                                       char* output, int n,
                                       size_t grad_value_size,
                                       DynamicGradMerger& merger,
                                      CommonFeatureValueAccessor& feature_value_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    uint32_t start = offset[i];
    uint32_t num = fea_num[i];
    int ori_index = index[start];
    float* out = (float*)(output + i * grad_value_size);
    float* in =
        (float*)(input + size_t(ori_index) * grad_value_size);
    merger.update_basic(out, in, feature_value_accessor);
    KeyType key = d_keys[i];
    if (key != 0) {
      for (int j = 1; j < num; ++j) {
        ori_index = index[start + j];
        in = (float*)(input + size_t(ori_index) * grad_value_size);
        merger.merge_basic(out, in, feature_value_accessor);
      }
    }
  }
}

template <typename KeyType>
__global__ void merge_gradients_embedx_kernel(const KeyType* d_keys,
                                       const uint32_t* offset,
                                       const uint32_t* fea_num,
                                       const uint32_t* index, const char* input,
                                       char* output, int n,
                                       size_t grad_dim,
                                       size_t grad_value_size,
                                       DynamicGradMerger& merger,
                                      CommonFeatureValueAccessor& feature_value_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    size_t value_idx = i / grad_dim;
    size_t field_idx = i % grad_dim;
    uint32_t start = offset[value_idx];
    uint32_t num = fea_num[value_idx];
    int ori_index = index[start];
    float* in = (float*)(input + size_t(ori_index) * grad_value_size);
    float* out = (float*)(output + value_idx * grad_value_size);
    merger.update_embedx(out, in, field_idx, feature_value_accessor);
    KeyType key = d_keys[value_idx];
    if (key != 0) {
      for (int j = 1; j < num; ++j) {
        int ori_index = index[start + j];
        float* in = (float*)(input + size_t(ori_index) * grad_value_size);
        merger.merge_embedx(out, in, field_idx, feature_value_accessor);
      }
    }
  }
}

__global__ void split_segments_kernel(
        const uint32_t* d_fea_num_info, size_t n,
        uint32_t* d_segments, uint32_t* d_segments_num,
        uint32_t segment_size) {
  const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx >= n) {
    return;
  }

  auto fea_num = d_fea_num_info[tx];
  auto seg_num = (uint32_t)((fea_num - 1) / segment_size + 1);
  d_segments[tx] = seg_num;
}

__global__ void expand_segments_kernel(
        const uint32_t* d_fea_num_info,
        const uint32_t* d_segments_offset, size_t n,
        uint32_t* d_segments_fea_num_info, uint32_t segment_size) {
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
__global__ void shrink_keys_kernel(
        const KeyType* d_keys, const uint32_t* d_segments_offset,
        KeyType* d_segments_keys, size_t n) {
  const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx >= n) {
    return;
  }

  d_segments_keys[tx] = d_keys[d_segments_offset[tx]];
}

template <typename T>
__global__ void dy_mf_fill_dvals_kernel(float* d_shard_vals, float* d_vals,
                                        T* idx, size_t len, size_t val_size,
                                       CommonFeatureValueAccessor feature_value_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint64_t new_offset = uint64_t(idx[i]) * val_size;
    float* cur = (float*)((char*)d_vals + new_offset);
    float* shard_val = (float*)((char*)d_shard_vals + uint64_t(i) * val_size);
    int mf_dim = int(shard_val[feature_value_accessor.common_feature_value.MfDimIndex()]);

    *(reinterpret_cast<uint64_t*>(cur + feature_value_accessor.common_feature_value.CpuPtrIndex())) =
      *(reinterpret_cast<uint64_t*>(shard_val + feature_value_accessor.common_feature_value.CpuPtrIndex()));
    cur[feature_value_accessor.common_feature_value.DeltaScoreIndex()] =
      shard_val[feature_value_accessor.common_feature_value.DeltaScoreIndex()];
    cur[feature_value_accessor.common_feature_value.ShowIndex()] =
      shard_val[feature_value_accessor.common_feature_value.ShowIndex()];
    cur[feature_value_accessor.common_feature_value.ClickIndex()] =
      shard_val[feature_value_accessor.common_feature_value.ClickIndex()];
    cur[feature_value_accessor.common_feature_value.EmbedWIndex()] =
      shard_val[feature_value_accessor.common_feature_value.EmbedWIndex()];
    for (int i = 0; i < feature_value_accessor.common_feature_value.EmbedDim(); i++) {
      cur[feature_value_accessor.common_feature_value.EmbedG2SumIndex() + i] = 
        shard_val[feature_value_accessor.common_feature_value.EmbedG2SumIndex() + i];
    }
    cur[feature_value_accessor.common_feature_value.SlotIndex()] =
      shard_val[feature_value_accessor.common_feature_value.SlotIndex()];
    cur[feature_value_accessor.common_feature_value.MfDimIndex()] = mf_dim;
    cur[feature_value_accessor.common_feature_value.MfSizeIndex()] =
      shard_val[feature_value_accessor.common_feature_value.MfSizeIndex()];

    for (int x = feature_value_accessor.common_feature_value.EmbedxG2SumIndex();
            x < int(feature_value_accessor.common_feature_value.Size(mf_dim) / sizeof(float)); x++){
      cur[x] = shard_val[x];
    }
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

template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::dy_mf_fill_shard_grads(
    KeyType* d_shard_keys, KeyType* d_keys, float* d_shard_grads,
    float* d_grads, T* idx, long long len, size_t grad_value_size,
    const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  dy_mf_fill_shard_grads_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, d_shard_grads, d_grads, idx, c_len,
      grad_value_size, feature_value_accessor_);
}

template <typename KeyType, typename StreamType>
void HeterCommKernel::merge_gradient(
    const KeyType* d_keys,
    const uint32_t* offset, const uint32_t* fea_num, const uint32_t* index,
    const char* input, char* output, int n, size_t grad_dim, size_t grad_value_size,
    DynamicGradMerger& merger, const StreamType& stream) {
  int grid_size1 = (n - 1) / block_size_ + 1;
  merge_gradients_basic_kernel<<<grid_size1, block_size_, 0, stream>>>(
      d_keys,
      offset, fea_num, index, input, output, n, grad_value_size, merger, feature_value_accessor_);
  if (grad_dim > 0) {
    int grid_size2 = (n * grad_dim - 1) / block_size_ + 1;
    merge_gradients_embedx_kernel<<<grid_size2, block_size_, 0, stream>>>(
            d_keys,
            offset, fea_num, index, input, output, n * grad_dim, grad_dim, grad_value_size, merger, feature_value_accessor_);
  }
}

template <typename T, typename StreamType>
void HeterCommKernel::dy_mf_fill_dvals(float* d_shard_vals, float* d_vals,
                                       T* idx, long long len, size_t val_size,
                                       const StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  size_t c_len = (size_t)len;
  dy_mf_fill_dvals_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_vals, d_vals, idx, c_len, val_size, feature_value_accessor_);
}

template <typename StreamType>
void HeterCommKernel::split_segments(const uint32_t* d_fea_num_info, size_t n,
        uint32_t* d_segments, uint32_t* d_segments_num, size_t segment_size, const StreamType& stream) {
  int grid_size = (n - 1) / block_size_ + 1;
  split_segments_kernel<<<grid_size, block_size_, 0, stream>>>(
          d_fea_num_info, n, d_segments, d_segments_num, segment_size);
}

template <typename StreamType>
void HeterCommKernel::expand_segments(const uint32_t* d_fea_num_info,
          const uint32_t* d_segments_offset, size_t n,
          uint32_t* d_segments_fea_num_info, uint32_t segment_size,
          const StreamType& stream) {
  int grid_size = (n - 1) / block_size_ + 1;
  expand_segments_kernel<<<grid_size, block_size_, 0, stream>>>(
          d_fea_num_info,
          d_segments_offset, n,
          d_segments_fea_num_info, segment_size);
}

template <typename KeyType, typename StreamType>
void HeterCommKernel::shrink_keys(const KeyType* d_keys, const uint32_t* d_segments_offset,
          KeyType* d_segments_keys, size_t n, const StreamType& stream) {
    int grid_size = (n - 1) / block_size_ + 1;
  shrink_keys_kernel<<<grid_size, block_size_, 0, stream>>>(
          d_keys, d_segments_offset, d_segments_keys, n);
}

template void HeterCommKernel::fill_idx<int, cudaStream_t>(
    int* idx, long long len, const cudaStream_t& stream);
template void HeterCommKernel::fill_idx<uint32_t, cudaStream_t>(
    uint32_t* idx, long long len, const cudaStream_t& stream);

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
    unsigned long, float, int, cudaStream_t>(
    unsigned long* d_shard_keys, unsigned long* d_keys,
    float* d_shard_grads,
    float* d_grads, int* idx, long long len,
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

template void HeterCommKernel::dy_mf_fill_shard_grads<
    unsigned long, int, cudaStream_t>(
    unsigned long* d_shard_keys, unsigned long* d_keys,
    float* d_shard_grads, float* d_grads, int* idx, long long len,
    size_t grad_value_size, const cudaStream_t& stream);

template void HeterCommKernel::merge_gradient<uint32_t, cudaStream_t>(
    const uint32_t* d_keys,
    const uint32_t* offset, const uint32_t* fea_num, const uint32_t* index,
    const char* input, char* output, int n, size_t grad_dim, size_t grad_value_size,
    DynamicGradMerger& merger_, const cudaStream_t& stream);

template void HeterCommKernel::merge_gradient<uint64_t, cudaStream_t>(
    const uint64_t* d_keys,
    const uint32_t* offset, const uint32_t* fea_num, const uint32_t* index,
    const char* input, char* output, int n, size_t grad_dim, size_t grad_value_size,
    DynamicGradMerger& merger_, const cudaStream_t& stream);

template void HeterCommKernel::dy_mf_fill_dvals<int, cudaStream_t>(
    float* d_shard_vals,
    float* d_vals, int* idx, long long len,
    size_t val_size, const cudaStream_t& stream);

template void HeterCommKernel::split_segments<cudaStream_t>(
    const uint32_t* d_fea_num_info, size_t n,
    uint32_t* d_segment, uint32_t* d_segments_num, size_t segment_size,
    const cudaStream_t& stream);

template void HeterCommKernel::expand_segments<cudaStream_t>(
    const uint32_t* d_fea_num_info,
    const uint32_t* d_segments_offset, size_t n,
    uint32_t* d_segments_fea_num_info, uint32_t segment_size,
    const cudaStream_t& stream);

template void HeterCommKernel::shrink_keys<uint32_t, cudaStream_t>(
        const uint32_t* d_keys, const uint32_t* d_segments_offset,
        uint32_t* d_segments_keys, size_t segment_num, const cudaStream_t& stream);

template void HeterCommKernel::shrink_keys<uint64_t, cudaStream_t>(
        const uint64_t* d_keys, const uint32_t* d_segments,
        uint64_t* d_segments_keys, size_t total_segment_num, const cudaStream_t& stream);
#endif

}  // namespace framework
}  // namespace paddle
#endif
