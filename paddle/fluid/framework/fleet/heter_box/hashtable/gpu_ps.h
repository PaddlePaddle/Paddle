/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "thrust/pair.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/memory/memory.h"
#include "cub/cub.cuh"
#include "hashtable.h"
#include "gpu_resource.h"
#ifdef PADDLE_WITH_PSLIB

namespace paddle {
namespace framework {

template <typename KeyType, typename ValType, typename GradType>
class GpuPs {
 public:
  GpuPs(size_t capacity, std::shared_ptr<HeterBoxResource> resource);
  ~GpuPs();
  GpuPs(const GpuPs&) = delete;
  GpuPs& operator=(const GpuPs&) = delete;

  void insert() {};
  void get(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len, size_t chunk_size, int stream_num);
  void dump();
  int log2i(unsigned x);

 private:
  
  using Table = HashTable<KeyType, ValType>;
  int block_size_{256};
  std::vector<Table*> tables_;
  std::shared_ptr<HeterBoxResource> resource_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
