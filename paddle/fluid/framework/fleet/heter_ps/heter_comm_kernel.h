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
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"

#if defined(PADDLE_WITH_CUDA)
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"
#endif

namespace paddle {
namespace framework {

class HeterCommKernel {
 public:
  HeterCommKernel() {}
  explicit HeterCommKernel(const int block_size) : block_size_(block_size) {}

  template <typename T, typename StreamType>
  void fill_idx(T* idx, long long len, const StreamType& stream);

  template <typename T, typename StreamType>
  void calc_shard_offset(T* idx, T* left, T* right, long long len,
                         int total_devs, const StreamType& stream);

  template <typename KeyType, typename T, typename StreamType>
  void calc_shard_index(KeyType* d_keys, long long len, T* shard_index,

                        int total_devs, const StreamType& stream);

  template <typename KeyType, typename T, typename StreamType>
  void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx,
                      long long len, const StreamType& stream);

  template <typename KeyType, typename GradType, typename T,
            typename StreamType>
  void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                        GradType* d_shard_grads, GradType* d_grads, T* idx,
                        long long len, const StreamType& stream);

  template <typename ValType, typename T, typename StreamType>
  void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx, long long len,
                  const StreamType& stream);

#if defined(PADDLE_WITH_XPU_KP)
  template <typename ValType, typename T, typename StreamType>
  void fill_dvals_with_bfid(ValType* d_all_vals, ValType* d_vals, T* idx, long long len,
                  const StreamType& stream);
#endif

#if defined(PADDLE_WITH_CUDA)
  template <typename KeyT, typename ValueT, typename StreamType>
  void sort_pairs(void* d_temp_storage, size_t& temp_storage_bytes,  // NOLINT
                  const KeyT* d_keys_in, KeyT* d_keys_out,
                  const ValueT* d_values_in, ValueT* d_values_out,
                  int num_items, int begin_bit = 0,

                  int end_bit = sizeof(KeyT) * 8, StreamType stream = NULL,
                  bool debug_synchronous = false);
#elif defined(PADDLE_WITH_XPU_KP)
  template <typename KeyT, typename ValueT, typename StreamType>
  void sort_pairs(const paddle::platform::Place& place,
                  void* d_temp_storage, size_t& temp_storage_bytes,  // NOLINT
                  const KeyT* d_keys_in, KeyT* d_keys_out,
                  const ValueT* d_values_in, ValueT* d_values_out,
                  int num_items, int begin_bit = 0,

                  int end_bit = sizeof(KeyT) * 8, StreamType stream = NULL,
                  bool debug_synchronous = false);
#endif

#if defined(PADDLE_WITH_CUDA)
  template <typename KeysInputIteratorT, typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT, typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT, typename StreamType>
  void reduce_by_key(void* d_temp_storage,
                     size_t& temp_storage_bytes,  // NOLINT
                     KeysInputIteratorT d_keys_in,
                     UniqueOutputIteratorT d_unique_out,
                     ValuesInputIteratorT d_values_in,
                     AggregatesOutputIteratorT d_aggregates_out,
                     NumRunsOutputIteratorT d_num_runs_out, int num_items,
                     StreamType stream = NULL, bool debug_synchronous = false);
#elif defined(PADDLE_WITH_XPU_KP)
  template <typename KeysInputIteratorT, typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT, typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT, typename StreamType>
  void reduce_by_key(const paddle::platform::Place& place,
                     void* d_temp_storage,
                     size_t& temp_storage_bytes,  // NOLINT
                     KeysInputIteratorT d_keys_in,
                     UniqueOutputIteratorT d_unique_out,
                     ValuesInputIteratorT d_values_in,
                     AggregatesOutputIteratorT d_aggregates_out,
                     NumRunsOutputIteratorT d_num_runs_out, int num_items,
                     StreamType stream = NULL, bool debug_synchronous = false);
#endif

  template <typename TID, typename ValueT, typename StreamType>
  void merge_grad(
    uint32_t first_fidseq_elem,
    TID* fidseq_grad_idxs,
    TID* fidseq_lods,
    int fidseq_lods_len,
    ValueT* d_vals_in,
    int len,
    ValueT* d_vals_out,
    const StreamType& stream);

#if defined(PADDLE_WITH_XPU_KP)

template <typename ValueT, typename StreamType>
void convert_feature_value_as_float(
    ValueT* d_vals_in,
    int len,
    bool to_float,
    const StreamType& stream);

template <typename ValueT, typename StreamType>
void convert_feature_push_value_as_float(
    ValueT* d_vals_in,
    int len,
    bool to_float,
    const StreamType& stream);

template <typename GradType, typename StreamType>
void sum_fidseq_add_grad(
    GradType* all_device_fidseq_grad_ptr,
    uint32_t all_device_fidseq_grad_len,
    const StreamType& stream,
    uint32_t bucket_num,
    GradType* out_fidseq_grad_compress_ptr);
#endif

 private:
  int block_size_{256};
};

}  // end namespace framework
}  // end namespace paddle
#endif
