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

#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>         // NOLINT
#include <unordered_map>  // NOLINT
#include <unordered_set>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/heter_service.pb.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#ifdef PADDLE_WITH_PSLIB
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

typedef std::function<int(const HeterRequest*, HeterResponse*)>
    HeterServiceHandler;
class DataFeed;

class HeterXpuService : public HeterService {
 public:
  HeterXpuService() {}
  virtual ~HeterXpuService() {}
  void service(::google::protobuf::RpcController* controller,
               const HeterRequest* request, HeterResponse* response,
               ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    int ret = 0;
    int cmd = request->cmd();
    auto itr = handler_map_.find(cmd);
    if (itr == handler_map_.end()) {
    } else {
      ret = itr->second(request, response);
    }
    // response->set_err_code(0);
    // response->set_err_msg("");
    if (ret != 0) {
      // response->set_err_code(-1);
      // response->set_err_msg("xpu service error");
    }
  }

  void RegisterServiceHandler(int cmd, HeterServiceHandler func) {
    VLOG(0) << "register heter service";
    handler_map_[cmd] = func;
  }

 private:
  std::unordered_map<int, HeterServiceHandler> handler_map_;
};

enum HeterTaskState { PULL_SPARSE, OP_RUN, XPU, OP_RUN_END, PUSH_GRAD, DONE };

class HeterTask {
 public:
  void Update() {
    if (state_ == PULL_SPARSE) {
      state_ = OP_RUN;
    } else if (state_ == OP_RUN) {
      state_ = XPU;
      // state_ = PUSH_GRAD;
      // state_ = PUSH_GRAD;
    } else if (state_ == XPU) {
      state_ = OP_RUN_END;
    } else if (state_ == OP_RUN_END) {
      state_ = PUSH_GRAD;
    } else if (state_ == PUSH_GRAD) {
      state_ = DONE;
    }
  }
  void Reset() {
    total_time = 0;
    read_time = 0;
    pack_time = 0;
    pull_sparse_local_time = 0;
    op_all_time = 0;
    xpu_op_time = 0;
    xpu_wait_time = 0;
    cpu_op_time = 0;
    collect_label_time = 0;
    fill_sparse_time = 0;
    push_sparse_time = 0;
    gpu_2_cpu_time = 0;
    cpu_2_gpu_time = 0;
    timeline.Reset();
  }
  void Show() {
    std::cout << "features size " << features_.size() << std::endl;
    for (size_t i = 0; i < features_.size(); ++i) {
      std::cout << "features[" << i << "] size " << features_[i].size()
                << std::endl;
    }
  }
  void PackTask(Scope* scope, int taskid, DataFeed* reader, int cur_batch,
                const ProgramDesc& program);
  void PackGpuTask(Scope* thread_scope, DataFeed* reader,
                   const ProgramDesc& program);

  Scope* scope_{nullptr};
  int taskid_;
  int cur_batch_;
  HeterTaskState state_;
  // cache
  std::map<uint64_t, std::vector<uint64_t>> features_;
  std::map<uint64_t, std::vector<float>> feature_labels_;
  std::map<uint64_t, std::vector<std::vector<float>>> feature_values_;
  std::map<uint64_t, std::vector<std::vector<float>>> feature_grads_;
  std::map<uint64_t, std::vector<uint64_t>> sparse_push_keys_;
  double total_time{0};
  double read_time{0};
  double pack_time{0};
  double pull_sparse_local_time{0};
  double op_all_time{0};
  double xpu_op_time{0};
  double xpu_wait_time{0};
  double cpu_op_time{0};
  double collect_label_time{0};
  double fill_sparse_time{0};
  double push_sparse_time{0};
  double gpu_2_cpu_time{0};
  double cpu_2_gpu_time{0};
  platform::Timer timeline;
};

template <class T>
class HeterObjectPool {
 public:
  HeterObjectPool() {}
  virtual ~HeterObjectPool(){};
  std::shared_ptr<T> Get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.empty()) {
      num_ += 1;
#ifdef PADDLE_WITH_CUDA
      VLOG(0) << "pool construct size: " << num_;
#endif
      return std::make_shared<T>();
    } else {
      auto ret = pool_.back();
      pool_.pop_back();
      return ret;
    }
  }
  void Push(std::shared_ptr<T> data) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(std::move(data));
  }
  int Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
  }
  std::shared_ptr<T>& GetElement(int i) { return pool_[i]; }

 private:
  std::vector<std::shared_ptr<T>> pool_;
  std::mutex mutex_;
  int num_{0};
};

struct BthreadMutextGuard {
  BthreadMutextGuard(bthread_mutex_t* rho) {
    mutex_ = rho;
    bthread_mutex_lock(mutex_);
  }
  ~BthreadMutextGuard() { bthread_mutex_unlock(mutex_); }
  bthread_mutex_t* mutex_;
};

template <class T>
class BtObjectPool {
 public:
  BtObjectPool() {
    bthread_mutex_init(&mutex_, NULL);
    bthread_cond_init(&cond_, NULL);
  }

  virtual ~BtObjectPool() {
    bthread_cond_destroy(&cond_);
    bthread_mutex_destroy(&mutex_);
  };

  std::shared_ptr<T> Get() {
    BthreadMutextGuard guard(&mutex_);
    while (pool_.empty()) {
      bthread_cond_wait(&cond_, &mutex_);
    }
    auto ret = pool_.back();
    pool_.pop_back();
    return ret;
  }

  void Push(std::shared_ptr<T> data) {
    BthreadMutextGuard guard(&mutex_);
    pool_.push_back(std::move(data));
    bthread_cond_signal(&cond_);
  }

  int Size() { return pool_.size(); }

  std::shared_ptr<T>& GetElement(int i) { return pool_[i]; }

 private:
  std::vector<std::shared_ptr<T>> pool_;
  bthread_mutex_t mutex_;
  bthread_cond_t cond_;
  int num_{0};
};

template <class K, class T>
struct HeterNode {
  K key;
  T value;
  HeterNode* prev;
  HeterNode* next;
};

template <class K, class T>
class HeterList {
 public:
  HeterList() : head_(new HeterNode<K, T>), tail_(new HeterNode<K, T>) {
    head_->prev = NULL;
    head_->next = tail_;
    tail_->prev = head_;
    tail_->next = NULL;
    size = 0;
    cap_ = 1e9;
  }

  ~HeterList() {
    delete head_;
    delete tail_;
  }

  void SetCap(int num) { cap_ = num; }

  bool TryPut(K& key, T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return size < cap_; });
    if (task_map_.find(key) != task_map_.end()) {
      // std::cout << "try put key=" << key << " false" << std::endl;
      task_map_.erase(key);
      return false;
    } else {
      HeterNode<K, T>* node = new HeterNode<K, T>;
      node->key = key;
      node->value = value;
      map_[node->key] = node;
      attach(node);
      // std::cout << "try put key=" << key << " true" << std::endl;
      return true;
    }
  }

  bool Put(K& key, T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return size < cap_; });
    HeterNode<K, T>* node = new HeterNode<K, T>;
    // std::cout << "put key=" << key << " true" << std::endl;
    node->key = key;
    node->value = value;
    map_[node->key] = node;
    attach(node);
    return true;
  }

  T TryGet(const K& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = map_.find(key);
    if (iter != map_.end()) {
      // std::cout << "try get key=" << key << " true" << std::endl;
      HeterNode<K, T>* node = iter->second;
      detach(node);
      cond_.notify_one();
      T ret = std::move(node->value);
      map_.erase(key);
      delete node;
      return ret;
    }
    task_map_.insert(key);
    // std::cout << "try get key=" << key << " false" << std::endl;
    return nullptr;
  }

  T Get(const K& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = map_.find(key);
    if (iter != map_.end()) {
      // std::cout << "get key=" << key << " true" << std::endl;
      HeterNode<K, T>* node = iter->second;
      detach(node);
      cond_.notify_one();
      T ret = std::move(node->value);
      map_.erase(key);
      delete node;
      return ret;
    }
    // std::cout << "get key=" << key << " false" << std::endl;
    return nullptr;
  }

  T Get() {
    std::lock_guard<std::mutex> lock(mutex_);
    HeterNode<K, T>* node = head_->next;
    if (node == tail_) {
      // std::cout << "get2 false" << std::endl;
      return nullptr;
    } else {
      detach(node);
      cond_.notify_one();
      T ret = std::move(node->value);
      map_.erase(node->key);
      // std::cout << "get2 key=" << node->key << " true" << std::endl;
      delete node;
      return ret;
    }
  }

  bool Empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return head_->next == tail_;
  }

  int Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return size;
  }

 private:
  void detach(HeterNode<K, T>* node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
    size--;
  }

  void attach(HeterNode<K, T>* node) {
    node->prev = head_;
    node->next = head_->next;
    head_->next->prev = node;
    head_->next = node;
    size++;
  }

 private:
  HeterNode<K, T>* head_;
  HeterNode<K, T>* tail_;
  std::unordered_map<K, HeterNode<K, T>*> map_;
  std::unordered_set<K> task_map_;
  std::mutex mutex_;
  std::condition_variable cond_;
  int cap_;
  int size;
};

}  // namespace framework
}  // namespace paddle
#endif
