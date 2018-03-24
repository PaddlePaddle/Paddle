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

#include "ThreadPool.h"  // ThreadPool in thrird party
#include "paddle/fluid/framework/details/ssa_graph_executor.h"

namespace paddle {
namespace framework {
class Scope;

namespace details {

class ThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  ThreadedSSAGraphExecutor(size_t num_threads, bool use_event,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<platform::Place> &places,
                           std::unique_ptr<SSAGraph> &&graph);

  // Run a SSAGraph by a thread pool
  // Use topological sort algorithm
  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;

  ~ThreadedSSAGraphExecutor() {}

 private:
  void RunOp(
      std::unordered_map<VarHandleBase *, std::atomic<bool>> &pending_vars,
      details::OpHandleBase *op);

 private:
  std::unique_ptr<::ThreadPool> pool_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_ctxs_;
  const bool use_event_;
  std::unique_ptr<platform::EnforceNotMet> exception_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
