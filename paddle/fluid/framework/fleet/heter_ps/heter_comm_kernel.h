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
namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
struct CustomGradMerger {
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
};
#elif defined(PADDLE_WITH_XPU)
struct CustomGradMerger {
  template <typename T>
  __device__ T operator()(const T& a, const T& b) const {
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
};
#endif

class HeterCommKernel {
 public:
  HeterCommKernel() {}
  HeterCommKernel(const int block_size) : block_size_(block_size) {}
  template <typename T, typename StreamType>
  void fill_idx(T* idx, long long len, const StreamType& stream);

  template <typename T, typename StreamType>
  void calc_shard_offset(T* idx, T* left, T* right, long long len,
                         int total_devs, const StreamType& stream);

  template <typename KeyType, typename T, typename StreamType>
  void calc_shard_index(KeyType* d_keys, long long len, T* shard_index,
                        int total_gpu, const StreamType& stream);

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

  CustomGradMerger merger_;

 private:
  int block_size_{256};
};

}  // end namespace framework
}  // end namespace paddle
#endif
