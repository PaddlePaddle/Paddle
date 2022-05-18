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

struct CustomGradMerger {
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

  template <typename T>
  __device__ __forceinline__ void copy_basic_field(T& output, const T& input) {
    output.slot = input.slot;
    output.show = input.show;
    output.clk = input.clk;
    output.mf_dim = input.mf_dim;
    output.lr_g = input.lr_g;
    for (int i = 0; i < output.mf_dim; ++i) {
      output.mf_g[i] = input.mf_g[i];
    }
  }
  template <typename T>
  __device__ __forceinline__ void add_basic_field(T& output, const T& input) {
    output.show += input.show;
    output.clk += input.clk;
    output.lr_g += input.lr_g;
    for (int i = 0; i < input.mf_dim; ++i) {
      output.mf_g[i] += input.mf_g[i];
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

  template <typename KeyT, typename ValueT, typename StreamType>
  void sort_pairs(void* d_temp_storage, size_t& temp_storage_bytes,  // NOLINT
                  const KeyT* d_keys_in, KeyT* d_keys_out,
                  const ValueT* d_values_in, ValueT* d_values_out,
                  int num_items, int begin_bit = 0,

                  int end_bit = sizeof(KeyT) * 8, StreamType stream = NULL,
                  bool debug_synchronous = false);

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

  template <typename KeyType, typename GradType, typename T,
            typename StreamType>
  void dy_mf_fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                              GradType* d_shard_grads, GradType* d_grads,
                              T* idx, long long len, size_t grad_value_size,
                              const StreamType& stream);

  template <typename StreamType>
  void merge_gradient(const uint32_t* offset, const uint32_t* fea_num,
                      const uint32_t* index, const char* input, char* output,
                      int n, size_t grad_value_size, CustomGradMerger& merger_,
                      const StreamType& stream);

  template <typename ValType, typename T, typename StreamType>
  void dy_mf_fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                        long long len, size_t val_size,
                        const StreamType& stream);

 private:
  int block_size_{256};
};

}  // end namespace framework
}  // end namespace paddle
#endif
