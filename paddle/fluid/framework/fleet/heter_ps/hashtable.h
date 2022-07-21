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
#include <fstream>
#include <sstream>
#include <chrono>

#ifdef PADDLE_WITH_PSLIB
#include "common_value.h"  // NOLINT
#endif

#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/memory/memory.h"
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
#elif defined(PADDLE_WITH_XPU_KP)
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#if defined(__xpu__)
#include <xpu/runtime.h>
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/math.h"
#include "xpu/kernel/simd.h"
#endif
#endif

#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"

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
  explicit XPUCacheArray(long long capacity, DevPlace& place) : capacity_(capacity), size_(0) {
    keys_auto_ptr_ = memory::Alloc(place, capacity_ * sizeof(KeyType));
    vals_auto_ptr_ = memory::Alloc(place, capacity_ * sizeof(ValType));
    keys_ = reinterpret_cast<KeyType*>(keys_auto_ptr_->ptr());
    vals_ = reinterpret_cast<ValType*>(vals_auto_ptr_->ptr());
    CPUPlace cpu_place = CPUPlace();
    cpu_keys_auto_ptr_ = memory::Alloc(cpu_place, capacity_ * sizeof(KeyType));
    cpu_vals_auto_ptr_ = memory::Alloc(cpu_place, capacity_ * sizeof(ValType));
    cpu_keys_ = reinterpret_cast<KeyType*>(cpu_keys_auto_ptr_->ptr());
    cpu_vals_ = reinterpret_cast<ValType*>(cpu_vals_auto_ptr_->ptr());
 }

  virtual ~XPUCacheArray() {
  }

  void print() {
    xpu_set_device(get_xpu_id());//TODO: logic to physics

    KeyType* keys = get_cpu_keys();
    ValType* vals = get_cpu_vals();

    sleep(1);
    auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* ptm = localtime(&now_time);
    char date[100] = {0};
    snprintf(date, 100, "%d%02d%02d%02d%02d%02d",
                    (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
                    (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);

    std::stringstream name_ss;
    name_ss << "cache_array-" << xpu_id_ << "." << date << ".dump";
    
    std::stringstream data_ss;
    data_ss << "xpu_id/num:" << xpu_id_ << "/" << xpu_num_ << "\n";
    data_ss << "size:" << size_ << "\n";
    for (long long i = 0; i < size_; i++) {
      data_ss << i << "\t" << keys[i] 
              << "\t" << vals[i] << "\n";
    }
    data_ss << "------------------------------------\n";

    std::ofstream ofs;
    ofs.open(name_ss.str(), std::ios::app);
    ofs << data_ss.str();
    ofs.close();    
  }

  void set_xpu_id(uint32_t xpu_id) { xpu_id_ = xpu_id; }
  void set_xpu_idx(uint32_t xpu_idx) { xpu_idx_ = xpu_idx; }
  void set_xpu_num(uint32_t xpu_num) { xpu_num_ = xpu_num; }
  void set_size(long long size) { size_ = size; }
  uint32_t get_xpu_id() {return xpu_id_;}
  uint32_t get_xpu_idx() {return xpu_idx_;}
  uint32_t get_xpu_num() {return xpu_num_;}
  KeyType* get_keys() {return keys_;}
  ValType* get_vals() {return vals_;}
  KeyType* get_cpu_keys() {
    xpu_memcpy(cpu_keys_, keys_, size_ * sizeof(KeyType), XPU_DEVICE_TO_HOST);
    return cpu_keys_;
  }
  ValType* get_cpu_vals() {
    xpu_memcpy(cpu_vals_, vals_, size_ * sizeof(ValType), XPU_DEVICE_TO_HOST);
    return cpu_vals_;
  }

  int prefetch(const int dev_id, XPUStream stream = NULL) { return 0; }
  size_t size() { return size_; }
  size_t capacity() { return capacity_; }

 private:
  long long capacity_;
  long long size_;
  memory::AllocationPtr keys_auto_ptr_, cpu_keys_auto_ptr_;
  memory::AllocationPtr vals_auto_ptr_, cpu_vals_auto_ptr_;
  KeyType* keys_;
  ValType* vals_;
  KeyType* cpu_keys_;
  ValType* cpu_vals_;
  uint32_t xpu_id_ = 0;
  uint32_t xpu_idx_ = 0;
  uint32_t xpu_num_ = 1;
};
#endif

template <typename KeyType, typename ValType>
class HashTable {
 public:
  explicit HashTable(size_t capacity, DevPlace& place);
  virtual ~HashTable();
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;

  template <typename StreamType>
  void insert(const paddle::platform::Place& place, const KeyType* d_keys, const ValType* d_vals, size_t len,
              StreamType stream);

  template <typename StreamType>
  void insert(const paddle::platform::Place& place, const KeyType* d_keys, size_t len, char* pool, size_t start_index,
              StreamType stream);

  void show();

#if defined(PADDLE_WITH_XPU_KP)
  void set_xpu_id(uint32_t xpu_id) { container_->set_xpu_id(xpu_id); }
  void set_xpu_idx(uint32_t xpu_idx) { container_->set_xpu_idx(xpu_idx); }
  void set_xpu_num(uint32_t xpu_num) { container_->set_xpu_num(xpu_num); }
#endif

  void set_sparse_sgd(const OptimizerConfig& optimizer_config);
  void set_embedx_sgd(const OptimizerConfig& optimizer_config);

  template <typename StreamType>
  void dump_to_cpu(int devid, StreamType stream);

#if defined(PADDLE_WITH_CUDA)

  template <typename GradType, typename Sgd, typename StreamType>
  void update(const KeyType* d_keys, const GradType* d_grads, size_t len,
              Sgd sgd, StreamType stream);

  template <typename Sgd, typename StreamType>
  void update(const KeyType* d_keys, const char* d_grads, size_t len, Sgd sgd,
              StreamType stream);

  template <typename StreamType>
  void get(const KeyType* d_keys, ValType* d_vals, size_t len,
           StreamType stream);

  template <typename StreamType>
  void get(const KeyType* d_keys, char* d_vals, size_t len, StreamType stream);

#elif defined(PADDLE_WITH_XPU_KP)
  template <typename GradType, typename StreamType>
  void update(const paddle::platform::Place& place, const KeyType* d_keys, const GradType* d_grads, size_t len,
              StreamType stream);

  template <typename StreamType>
  void update(const paddle::platform::Place& place, const KeyType* d_keys, const char* d_grads, size_t len,
              StreamType stream);

  template <typename StreamType>
  void get(const paddle::platform::Place& place, const KeyType* d_keys, ValType* d_vals, size_t len,
           StreamType stream);

  template <typename StreamType>
  void get(const paddle::platform::Place& place, const KeyType* d_keys, char* d_vals, size_t len, StreamType stream);

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
  memory::AllocationPtr OptimizerConfigAutoPtr_;
#endif
  OptimizerConfig* device_optimizer_config_;
  OptimizerConfig host_optimizer_config_;

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
