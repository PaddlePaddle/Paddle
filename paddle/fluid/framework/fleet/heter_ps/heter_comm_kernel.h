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
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"

#if defined(PADDLE_WITH_CUDA)
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"
#endif

namespace paddle {
namespace framework {

struct DynamicGradMerger {
  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__ __device__ T
  operator()(const T& a, const T& b) const {
    T out;
    out.slot = a.slot;
    out.mf_dim = a.mf_dim;
    out.show = a.show + b.show;
    out.clk = a.clk + b.clk;
    out.lr_g = a.lr_g + b.lr_g;

    return out;
  }

  template <typename GPUAccessor>
  __device__ __forceinline__ void update_one(float* output,
                                             const float* input,
                                             GPUAccessor& gpu_accessor) {
    gpu_accessor.PushValueFill(output, input);
  }

  template <typename GPUAccessor>
  __device__ __forceinline__ void merge_one(float* output,
                                            const float* input,
                                            GPUAccessor& gpu_accessor) {
    gpu_accessor.MergePushValue(output, input);
  }

  template <typename GPUAccessor>
  __device__ __forceinline__ void update_basic(float* output,
                                               const float* input,
                                               GPUAccessor& fv_accessor) {
    fv_accessor.PushValueFillBasic(output, input);
  }

  template <typename GPUAccessor>
  __device__ __forceinline__ void merge_basic(float* output,
                                              const float* input,
                                              GPUAccessor& fv_accessor) {
    fv_accessor.MergePushValueBasic(output, input);
  }

  template <typename GPUAccessor>
  __device__ __forceinline__ void update_embedx(float* output,
                                                const float* input,
                                                size_t embedx_idx,
                                                GPUAccessor& fv_accessor) {
    if (embedx_idx < output[fv_accessor.common_push_value.MfDimIndex()]) {
      output[fv_accessor.common_push_value.EmbedxGIndex() + embedx_idx] =
          input[fv_accessor.common_push_value.EmbedxGIndex() + embedx_idx];
    }
  }

  template <typename GPUAccessor>
  __device__ __forceinline__ void merge_embedx(float* output,
                                               const float* input,
                                               size_t embedx_idx,
                                               GPUAccessor& fv_accessor) {
    if (embedx_idx < output[fv_accessor.common_push_value.MfDimIndex()]) {
      output[fv_accessor.common_push_value.EmbedxGIndex() + embedx_idx] +=
          input[fv_accessor.common_push_value.EmbedxGIndex() + embedx_idx];
    }
  }
};

class HeterCommKernel {
 public:
  HeterCommKernel() {}
  explicit HeterCommKernel(const int block_size) : block_size_(block_size) {}

  template <typename T, typename StreamType>
  void fill_idx(T* idx, long long len, const StreamType& stream);

  template <typename T, typename StreamType>
  void calc_shard_offset(T* idx,
                         T* left,
                         T* right,
                         long long len,
                         int total_devs,
                         const StreamType& stream);

  template <typename KeyType, typename T, typename StreamType>
  void calc_shard_index(KeyType* d_keys,
                        long long len,
                        T* shard_index,

                        int total_devs,
                        const StreamType& stream);

  template <typename KeyType, typename T, typename StreamType>
  void fill_shard_key(KeyType* d_shard_keys,
                      KeyType* d_keys,
                      T* idx,
                      long long len,
                      const StreamType& stream);

  template <typename KeyType,
            typename GradType,
            typename T,
            typename StreamType>
  void fill_shard_grads(KeyType* d_shard_keys,
                        KeyType* d_keys,
                        GradType* d_shard_grads,
                        GradType* d_grads,
                        T* idx,
                        long long len,
                        const StreamType& stream);

  template <typename ValType, typename T, typename StreamType>
  void fill_dvals(ValType* d_shard_vals,
                  ValType* d_vals,
                  T* idx,
                  long long len,
                  const StreamType& stream);

  template <typename KeyT, typename ValueT, typename StreamType>
  void sort_pairs(void* d_temp_storage,
                  size_t& temp_storage_bytes,  // NOLINT
                  const KeyT* d_keys_in,
                  KeyT* d_keys_out,
                  const ValueT* d_values_in,
                  ValueT* d_values_out,
                  int num_items,
                  int begin_bit = 0,

                  int end_bit = sizeof(KeyT) * 8,
                  StreamType stream = NULL,
                  bool debug_synchronous = false);

  template <typename KeysInputIteratorT,
            typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT,
            typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT,
            typename StreamType>
  void reduce_by_key(void* d_temp_storage,
                     size_t& temp_storage_bytes,  // NOLINT
                     KeysInputIteratorT d_keys_in,
                     UniqueOutputIteratorT d_unique_out,
                     ValuesInputIteratorT d_values_in,
                     AggregatesOutputIteratorT d_aggregates_out,
                     NumRunsOutputIteratorT d_num_runs_out,
                     int num_items,

                     StreamType stream = NULL,
                     bool debug_synchronous = false);

  template <typename KeyType,
            typename T,
            typename StreamType,
            typename GPUAccessor>
  void dy_mf_fill_shard_grads(KeyType* d_shard_keys,
                              KeyType* d_keys,
                              float* d_shard_grads,
                              float* d_grads,
                              T* idx,
                              long long len,
                              size_t grad_value_size,
                              const StreamType& stream,
                              GPUAccessor& gpu_accessor);

  template <typename KeyType, typename StreamType, typename GPUAccessor>
  void merge_gradient(const KeyType* d_shard_keys,
                      const uint32_t* offset,
                      const uint32_t* fea_num,
                      const uint32_t* index,
                      const char* input,
                      char* output,
                      int n,
                      size_t grad_dim,
                      size_t grad_value_size,
                      DynamicGradMerger& merger,
                      const StreamType& stream,
                      GPUAccessor& gpu_accessor);

  template <typename T, typename StreamType>
  void dy_mf_fill_dvals(float* d_shard_vals,
                        float* d_vals,
                        T* idx,
                        long long len,
                        size_t val_size,
                        const StreamType& stream);

  template <typename StreamType>
  void split_segments(const uint32_t* d_fea_num_info,
                      size_t len,
                      uint32_t* d_segments,
                      uint32_t* d_segments_num,
                      size_t segment_size,
                      const StreamType& stream);

  template <typename StreamType>
  void expand_segments(const uint32_t* d_fea_num_info,
                       const uint32_t* d_segments_offset,
                       size_t segments_num,
                       uint32_t* d_segments_fea_num_info,
                       uint32_t segment_size,
                       const StreamType& stream);

  template <typename KeyType, typename StreamType>
  void shrink_keys(const KeyType* d_keys,
                   const uint32_t* d_segments_offset,
                   KeyType* d_segments_keys,
                   size_t segments_num,
                   const StreamType& stream);

  template <typename KeyType, typename StreamType>
  void fill_restore_idx(bool filter_zero,
                        const size_t total_num,
                        const size_t merge_size,
                        const KeyType* d_keys,
                        const uint32_t* d_sorted_idx,
                        const uint32_t* d_offset,
                        const uint32_t* d_merged_cnts,
                        uint32_t* d_restore_idx,
                        const StreamType& stream);

  template <typename KeyType, typename StreamType>
  void unpack_merged_vals(size_t n,
                          const KeyType* d_keys,
                          const void* d_merged_vals,
                          const uint32_t* d_restore_idx,
                          void* d_vals,
                          size_t val_size,
                          const StreamType& stream);

 private:
  int block_size_{256};
};

}  // end namespace framework
}  // end namespace paddle
#endif
