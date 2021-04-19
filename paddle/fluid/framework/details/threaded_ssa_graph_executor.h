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

#include <ThreadPool.h>  // ThreadPool in thrird party

#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
class Scope;

namespace details {

struct OpDependentData {
  std::unordered_map<OpHandleBase *, size_t> pending_ops_;
  std::unordered_set<VarHandleBase *> pending_vars_;
  std::unordered_set<OpHandleBase *> ready_ops_;
  size_t num_ops_{0};
};

class ThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  ThreadedSSAGraphExecutor(const ExecutionStrategy &strategy,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<Scope *> &local_exec_scopes,
                           const std::vector<platform::Place> &places,
                           ir::Graph *graph);

  const ir::Graph &Graph() const override { return *graph_; }
  // Run a SSAGraph by a thread pool
  // Use topological sort algorithm
  FetchResultType Run(const std::vector<std::string> &fetch_tensors,
                      bool return_merged) override;

  ~ThreadedSSAGraphExecutor() final = default;

 private:
  inline FetchResultType RunImpl(const std::vector<std::string> &fetch_tensors,
                                 bool return_merged);
  void RunOp(const std::shared_ptr<BlockingQueue<VarHandleBase *>> &ready_var_q,
             details::OpHandleBase *op);

 private:
  // Note(zcd): the ThreadPool should be placed last so that ThreadPool should
  // be destroyed first.
  ir::Graph *graph_;
  std::vector<Scope *> local_scopes_;
  std::vector<Scope *> local_exec_scopes_;

  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_ctxs_;
  ExceptionHolder exception_holder_;
  std::unique_ptr<OpDependentData> op_deps_;
  std::future<std::unique_ptr<OpDependentData>> op_deps_futures_;
  ExecutionStrategy strategy_;
  // use std::list because clear(), push_back, and for_each are O(1)
  std::list<std::future<void>> run_op_futures_;
  ::ThreadPool prepare_pool_;
  std::unique_ptr<::ThreadPool> pool_;
  std::vector<OpHandleBase *> traced_ops_;

  void InsertPendingOp(std::unordered_map<OpHandleBase *, size_t> *pending_ops,
                       OpHandleBase *op_instance) const;

  void InsertPendingVar(std::unordered_set<VarHandleBase *> *pending_vars,
                        std::unordered_set<VarHandleBase *> *ready_vars,
                        VarHandleBase *var) const;

  void InsertFetchOps(const std::vector<std::string> &fetch_tensors,
                      std::vector<OpHandleBase *> *fetch_ops,
                      std::unordered_set<VarHandleBase *> *fetch_dependencies,
                      std::unordered_set<OpHandleBase *> *ready_ops,
                      std::unordered_map<OpHandleBase *, size_t> *pending_ops,
                      std::unordered_set<VarHandleBase *> *pending_vars,
                      FetchResultType *fetch_data, bool return_merged);

  void PrepareOpDeps();

  void CopyOpDeps();

  inline void RecordOps(OpHandleBase *op);

  inline void ExecutionFinal(std::vector<OpHandleBase *> *fetch_ops);

  inline bool RunOpSync(OpHandleBase *op);

  bool RunTracedOps(const std::vector<OpHandleBase *> &traced_ops);
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
