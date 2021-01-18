// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <ThreadPool.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"

#if defined(PADDLE_WITH_XPU)
namespace paddle {
namespace framework {
class Scope;
namespace details {

struct RunningItem {
  std::atomic<int> dep_num;
  OpHandleBase *op;
};

class OpHandleBase;
class BindThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  BindThreadedSSAGraphExecutor(const ExecutionStrategy &strategy,
                               const std::vector<Scope *> &local_scopes,
                               const std::vector<Scope *> &local_exec_scopes,
                               const std::vector<platform::Place> &places,
                               ir::Graph *graph);
  // FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;
  // Run a SSAGraph by a thread pool
  // Use topological sort algorithm
  FetchResultType Run(const std::vector<std::string> &fetch_tensors,
                      bool return_merged) override;
  const ir::Graph &Graph() const override;

 private:
  FetchResultType RunMainStream(const std::vector<std::string> &fetch_tensors,
                                bool return_merged);

  // Note(zcd): the ThreadPool should be placed last so that ThreadPool should
  // be destroyed first.
  ExecutionStrategy strategy_;
  std::vector<Scope *> local_scopes_;
  std::vector<Scope *> local_exec_scopes_;
  std::vector<platform::Place> places_;
  ir::Graph *graph_;

  std::unordered_map<OpHandleBase *, int> op_deps_;
  std::unordered_map<int, int> place_to_index_;
  std::vector<OpHandleBase *> bootstrap_ops_;

  std::unique_ptr<int[]> stream_op_count_;

  std::future<
      std::unique_ptr<std::unordered_map<OpHandleBase *, struct RunningItem>>>
      atomic_op_deps_;
  ExceptionHolder exception_;

  std::vector<std::unique_ptr<::ThreadPool>> pool_;
  ::ThreadPool prepare_pool_;
  ::ThreadPool multi_device_op_pool_;

  void RunOpAsyncMainStream(
      OpHandleBase *op,
      std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
      std::shared_ptr<BlockingQueue<OpHandleBase *>> ready_ops, int index);

  void RunMultiDeviceOpAsync(
      OpHandleBase *op,
      std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
      std::shared_ptr<BlockingQueue<OpHandleBase *>> ready_ops);

  void PrepareAtomicOpDeps();

  int get_pool_thread_index(int device_id);

  inline void ExecutionFinal(std::vector<OpHandleBase *> *fetch_ops);

  void InsertFetchOps(
      const std::vector<std::string> &fetch_tensors, FetchResultType *fetches,
      std::unordered_map<std::string, std::vector<VarHandleBase *>>
          *fetched_vars,
      std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
      std::vector<OpHandleBase *> *fetch_ops,
      std::vector<OpHandleBase *> *ready_fetch_ops, bool return_merged);
};
}  // namespace details
}  // namespace framework
}  // namespace paddle

#endif
