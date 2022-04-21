/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <limits>
#include <memory>
#include <vector>

#ifdef PADDLE_WITH_PSLIB
#include "common_value.h"  // NOLINT
#endif

#if defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#endif
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/phi/core/utils/rw_lock.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/framework/fleet/heter_ps/cudf/concurrent_unordered_map.cuh.h"
#include "paddle/fluid/framework/fleet/heter_ps/mem_pool.h"
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#include "thrust/pair.h"
#elif defined(__xpu__)
#include <xpu/runtime.h>
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/math.h"
#include "xpu/kernel/simd.h"
#endif

#if defined(PADDLE_WITH_XPU_KP)
#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"
#endif

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
template <typename KeyType, typename ValType>
class TableContainer
    : public concurrent_unordered_map<KeyType, ValType,
                                      std::numeric_limits<KeyType>::max()> {
 public:
  TableContainer(size_t capacity)
      : concurrent_unordered_map<KeyType, ValType,
                                 std::numeric_limits<KeyType>::max()>(
            capacity, ValType()) {}
};
#elif defined(PADDLE_WITH_XPU_KP)
template <typename KeyType, typename ValType>
class XPUCacheArray {
 public:
  explicit XPUCacheArray(long long capacity) : capacity_(capacity), size_(0) {
    xpu_malloc(reinterpret_cast<void**>(&keys), capacity_ * sizeof(KeyType));
    xpu_malloc(reinterpret_cast<void**>(&vals), capacity_ * sizeof(ValType));
  }

  virtual ~XPUCacheArray() {
    xpu_free(keys);
    xpu_free(vals);
  }

  void print() {}

#if defined(__xpu__)
  __device__ ValType* find(const KeyType& key) {
    for (int i = 0; i < size_; i++) {
      if (keys[i] == key) return &vals[i];
    }
    return NULL;
  }
  __device__ bool insert(const KeyType& key, const ValType& val) {
    // # NOTE(zhangminxu): we set the capacity larger than the feasign number of
    // one batch
    if (size_ == capacity_) {
      return false;
    } else {
      keys[size_] = key;
      vals[size_] = val;
      size_++;
      return true;
    }
  }
#endif

  int prefetch(const int dev_id, XPUStream stream = NULL) { return 0; }
  size_t size() { return size_; }

 private:
  long long capacity_;
  long long size_;
  KeyType* keys;
  ValType* vals;
};
#endif

template <typename KeyType, typename ValType>
class HashTable {
 public:
  explicit HashTable(size_t capacity);
  virtual ~HashTable();
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;

  template <typename StreamType>
  void insert(const KeyType* d_keys, const ValType* d_vals, size_t len,
              StreamType stream);

  template <typename StreamType>
  void insert(const KeyType* d_keys, size_t len, char* pool, size_t start_index,
              StreamType stream);

  template <typename StreamType>
  void get(const KeyType* d_keys, ValType* d_vals, size_t len,
           StreamType stream);

  template <typename StreamType>
  void get(const KeyType* d_keys, char* d_vals, size_t len, StreamType stream);

  void show();

#if defined(PADDLE_WITH_XPU_KP)
  void set_sparse_sgd(const OptimizerConfig& optimizer_config);
  void set_embedx_sgd(const OptimizerConfig& optimizer_config);
#endif

  template <typename StreamType>
  void dump_to_cpu(int devid, StreamType stream);

#if defined(PADDLE_WITH_CUDA)

  template <typename GradType, typename Sgd, typename StreamType>
  void update(const KeyType* d_keys, const GradType* d_grads, size_t len,
              Sgd sgd, StreamType stream);

  template <typename Sgd, typename StreamType>
  void update(const KeyType* d_keys, const char* d_grads, size_t len, Sgd sgd,
              StreamType stream);

#elif defined(PADDLE_WITH_XPU_KP)
  template <typename GradType, typename StreamType>
  void update(const KeyType* d_keys, const GradType* d_grads, size_t len,
              StreamType stream);

  template <typename StreamType>
  void update(const KeyType* d_keys, const char* d_grads, size_t len,
              StreamType stream);

#endif

  int size() { return container_->size(); }

  void set_feature_value_size(size_t pull_feature_value_size,
                              size_t push_grad_value_size) {
    pull_feature_value_size_ = pull_feature_value_size;
    push_grad_value_size_ = push_grad_value_size;
    VLOG(3) << "hashtable set pull value size: " << pull_feature_value_size_
            << " push value size: " << push_grad_value_size_;
  }

  std::unique_ptr<phi::RWLock> rwlock_{nullptr};

 private:
#if defined(PADDLE_WITH_CUDA)
  TableContainer<KeyType, ValType>* container_;
#elif defined(PADDLE_WITH_XPU_KP)
  XPUCacheArray<KeyType, ValType>* container_;
  OptimizerConfig* xpu_optimizer_config_;
  OptimizerConfig cpu_optimizer_config_;
#endif
  int BLOCK_SIZE_{256};
  float LOAD_FACTOR{0.75f};
  size_t capacity_;
  size_t max_mf_dim_ = 8;
  size_t pull_feature_value_size_;
  size_t push_grad_value_size_;
};
}  // end namespace framework
}  // end namespace paddle
#endif
