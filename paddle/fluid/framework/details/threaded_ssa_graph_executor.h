//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <deque>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <functional>
#include "ThreadPool.h"  // ThreadPool in thrird party

#include "paddle/fluid/framework/details/ssa_graph_executor.h"

#include "paddle/fluid/framework/details/fetch_op_handle.h"

namespace paddle {
namespace framework {
class Scope;

namespace details {

template <typename T>
class BlockingQueue {
 public:
  BlockingQueue() {}

  void Push(const T &item) {
    {
      std::lock_guard<std::mutex> g(mutex_);
      q_.emplace_back(item);
    }
    cv_.notify_one();
  }

  template <typename U>
  void Extend(const U &items) {
    {
      std::lock_guard<std::mutex> g(mutex_);
      for (auto &item : items) {
        q_.emplace_back(item);
      }
    }
    cv_.notify_all();
  }

  std::deque<T> PopAll(size_t ms, bool *timeout) {
    auto time =
        std::chrono::system_clock::now() + std::chrono::milliseconds(ms);
    std::unique_lock<std::mutex> lock(mutex_);
    *timeout = !cv_.wait_until(lock, time, [this] { return !q_.empty(); });
    std::deque<T> ret;
    if (!*timeout) {
      std::swap(ret, q_);
    }
    return ret;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> q_;
};

class OpHandleBase;
class VarHandleBase;

class ThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  ThreadedSSAGraphExecutor(size_t num_threads, bool use_event,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<platform::Place> &places,
                           std::unique_ptr<SSAGraph> &&graph,
                           bool allow_op_delay);

  // Run a SSAGraph by a thread pool
  // Use topological sort algorithm
  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;

  ~ThreadedSSAGraphExecutor() {}

 private:
  struct StoredContext {
    std::unordered_map<OpHandleBase *, size_t> pending_ops_;
    std::unordered_set<VarHandleBase *> pending_vars_;
    std::unordered_set<OpHandleBase *> ready_ops_;
    std::vector<VarHandleBase *> ready_vars_;

    std::vector<std::unique_ptr<FetchOpHandle>> fetch_ops_;
    std::vector<LoDTensor> fetched_tensors_;
    std::vector<std::unique_ptr<VarHandleBase>> fetch_dependencies_;
  };

  class RunContext {
   public:
    explicit RunContext(StoredContext *stored_context);

    std::unordered_map<OpHandleBase *, size_t> pending_ops_;
    std::unordered_set<VarHandleBase *> pending_vars_;
    std::unordered_set<OpHandleBase *> ready_ops_;
    std::vector<VarHandleBase *> ready_vars_;

    std::vector<LoDTensor> FetchedResult() const;

    ~RunContext();

   private:
    StoredContext *stored_context_;
  };

  void RunOp(BlockingQueue<VarHandleBase *> *ready_var_q,
             details::OpHandleBase *op);

  void RunDelayedOps(const std::unordered_set<OpHandleBase *> &delayed_ops);

  RunContext PrepareOrGetContext(const std::vector<std::string> &fetch_tensors);

 private:
  std::unique_ptr<::ThreadPool> pool_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_ctxs_;
  const bool use_event_;
  std::unique_ptr<platform::EnforceNotMet> exception_;
  std::atomic<int> running_ops_;
  bool allow_op_delay_;
  struct FetchNameHash {
    size_t operator()(const std::vector<std::string> &names) const {
      std::hash<std::string> s_hash;
      std::hash<size_t> i_hash;

      size_t res = 0;
      for (auto &item : names) {
        res += s_hash(item);
      }

      return i_hash(res);
    }
  };

  std::unordered_map<std::vector<std::string>, StoredContext, FetchNameHash>
      contexts_;

  size_t computation_count_{0};
  size_t max_async_computation{100};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
