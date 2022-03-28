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
namespace paddle {
namespace framework {

class HeterCommKernel {
 public:
  HeterCommKernel() {}
  HeterCommKernle(const int block_size) : block_size_(block_size) {}
  template <typename T, typename StreamType>
  void fill_idx(T* idx, size_t len, const StreamType& stream);

  template <typename T, typename StreamType>
  void calc_shard_offset(T* idx, T* left, T* right, size_t len,
                         size_t total_devs, const StreamType& stream);

  template <typename KeyType, typename T, typename StreamType>
  void calc_shard_index(KeyType* d_keys, size_t len, T* shard_index,
                        int total_gpu, const StreamType& stream);

  template <typename KeyType, typename T, typename StreamType>
  void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx,
                      size_t len, const StreamType& stream);

  template <typename KeyType, typename GradType, typename T,
            typename StreamType>
  void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                        GradType* d_shard_grads, GradType* d_grads, T* idx,
                        size_t len, const StreamType& stream);

  template <typename ValType, typename T, typename StreamType>
  void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx, size_t len,
                  const StreamType& stream);

 private:
  int block_size_{256};
};

}  // end namespace framework
}  // end namespace paddle
#endif
