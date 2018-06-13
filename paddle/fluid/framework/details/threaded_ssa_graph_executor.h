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
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"

namespace paddle {
namespace framework {
class Scope;

namespace details {

class ThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  ThreadedSSAGraphExecutor(const ExecutionStrategy &strategy,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<platform::Place> &places,
                           std::unique_ptr<SSAGraph> &&graph);

  // Run a SSAGraph by a thread pool
  // Use topological sort algorithm
  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;

  ~ThreadedSSAGraphExecutor() {}

 private:
  void RunOp(BlockingQueue<VarHandleBase *> *ready_var_q,
             details::OpHandleBase *op);

 private:
  std::unique_ptr<SSAGraph> graph_;
  std::unique_ptr<::ThreadPool> pool_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_ctxs_;
  std::unique_ptr<platform::EnforceNotMet> exception_;
  std::atomic<int> running_ops_;

  void InsertPendingOp(std::unordered_map<OpHandleBase *, size_t> *pending_ops,
                       OpHandleBase *op_instance) const;

  void InsertPendingVar(std::unordered_set<VarHandleBase *> *pending_vars,
                        BlockingQueue<VarHandleBase *> *ready_vars,
                        VarHandleBase *var) const;

  void InsertFetchOps(
      const std::vector<std::string> &fetch_tensors,
      std::vector<std::unique_ptr<FetchOpHandle>> *fetch_ops,
      std::unordered_set<std::unique_ptr<VarHandleBase>> *fetch_dependencies,
      std::unordered_map<OpHandleBase *, size_t> *pending_ops,
      std::unordered_set<VarHandleBase *> *pending_vars,
      BlockingQueue<VarHandleBase *> *ready_vars, FeedFetchList *fetch_data);

 private:
  ExecutionStrategy strategy_;
};

class FasterSSAGraphExecutor : public SSAGraphExecutor {
 public:
  FasterSSAGraphExecutor(const ExecutionStrategy &strategy,
                         const std::vector<Scope *> &local_scopes,
                         const std::vector<platform::Place> &places,
                         std::unique_ptr<SSAGraph> &&graph);

  ~FasterSSAGraphExecutor();
  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;

 private:
  void InsertFetchOps(
      const std::vector<std::string> &fetch_tensors,
      std::vector<std::unique_ptr<FetchOpHandle>> *fetch_ops,
      std::unordered_set<std::unique_ptr<VarHandleBase>> *fetch_dependencies,
      std::unordered_map<OpHandleBase *, std::atomic<size_t>> *pending_ops,
      FeedFetchList *fetch_data);

  struct JobItem {
    OpHandleBase *op_{nullptr};
    std::unordered_map<OpHandleBase *, std::atomic<size_t>> *pending_ops_{
        nullptr};
    size_t *op_counter_{nullptr};
    std::mutex *op_counter_mtx_{nullptr};
    std::condition_variable *op_counter_cv_{nullptr};

    JobItem() {}
    JobItem(
        OpHandleBase *op,
        std::unordered_map<OpHandleBase *, std::atomic<size_t>> *pending_ops,
        size_t *op_counter, std::mutex *op_counter_mtx,
        std::condition_variable *op_counter_cv)
        : op_(op),
          pending_ops_(pending_ops),
          op_counter_(op_counter),
          op_counter_mtx_(op_counter_mtx),
          op_counter_cv_(op_counter_cv) {}
  };

  void ThreadFunc();

  std::unique_ptr<SSAGraph> graph_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_ctxs_;
  BlockingQueue<JobItem> jobs_;
  std::vector<std::thread> threads_;
  ExecutionStrategy strategy_;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
