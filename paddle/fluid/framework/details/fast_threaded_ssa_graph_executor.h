// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>
#include "ThreadPool.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"

namespace paddle {
namespace framework {
class Scope;
namespace details {

class OpHandleBase;
class FastThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  FastThreadedSSAGraphExecutor(const ExecutionStrategy &strategy,
                               const std::vector<Scope *> &local_scopes,
                               const std::vector<platform::Place> &places,
                               std::unique_ptr<ir::Graph> &&graph);
  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;
  const ir::Graph &Graph() const override;

 private:
  ExecutionStrategy strategy_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  std::unique_ptr<ir::Graph> graph_;

  std::unordered_map<OpHandleBase *, int> op_deps_;
  std::vector<OpHandleBase *> bootstrap_ops_;

  ::ThreadPool pool_;
  platform::DeviceContextPool fetch_ctxs_;
  std::atomic<int> remaining_;

  void RunOpAsync(std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
                  OpHandleBase *op,
                  const std::shared_ptr<BlockingQueue<size_t>> &complete_q);

  void PrepareAtomicOpDeps();

  std::future<
      std::unique_ptr<std::unordered_map<OpHandleBase *, std::atomic<int>>>>
      atomic_op_deps_;
  ExceptionHolder exception_;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
