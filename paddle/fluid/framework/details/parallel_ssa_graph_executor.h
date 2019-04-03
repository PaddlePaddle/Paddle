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
#include "paddle/fluid/framework/details/fast_threaded_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

class ParallelSSAGraphExecutor : public SSAGraphExecutor {
 public:
  ParallelSSAGraphExecutor(const ExecutionStrategy &strategy,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<platform::Place> &places,
                           ir::Graph *graph);
  ~ParallelSSAGraphExecutor() final = default;

  const ir::Graph &Graph() const override { return *graphs_[0]; }

  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;

 private:
  std::vector<std::unique_ptr<ir::Graph>> SeparateMultiDevicesGraph(
      ir::Graph *graph);

  ExecutionStrategy strategy_;
  std::vector<Scope *> local_scopes_;
  std::unique_ptr<::ThreadPool> pool_{nullptr};
  std::vector<platform::Place> places_;
  std::vector<std::unique_ptr<ir::Graph>> graphs_;

  std::vector<std::unique_ptr<details::FastThreadedSSAGraphExecutor>>
      executors_;
  ExceptionHolder exception_holder_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
