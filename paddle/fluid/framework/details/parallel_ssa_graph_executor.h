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

#include <memory>
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
  enum FeedStatus {
    kNone = 0,    // No feed
    kHasFeed = 1  // Has feed
  };

 public:
  ParallelSSAGraphExecutor(const ExecutionStrategy &strategy,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<Scope *> &local_exec_scopes,
                           const std::vector<platform::Place> &places,
                           ir::Graph *graph);

  ParallelSSAGraphExecutor(const ExecutionStrategy &strategy,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<Scope *> &local_exec_scopes,
                           const std::vector<platform::Place> &places,
                           std::vector<std::unique_ptr<ir::Graph>> graphs);

  ~ParallelSSAGraphExecutor() final = default;

  const ir::Graph &Graph() const override { return *graphs_[0]; }

  std::vector<ir::Graph *> Graphs();

  FetchResultType Run(const std::vector<std::string> &fetch_tensors,
                      bool return_merged) override;

  void SetHasFeed(size_t dev_idx, bool has_feed) {
    feed_status_[dev_idx] = has_feed ? FeedStatus::kHasFeed : FeedStatus::kNone;
  }

  void EnablePartialFeedSupport() { support_partial_feed_ = true; }

  bool SupportPartialFeed() const { return support_partial_feed_; }

 private:
  ExecutionStrategy strategy_;
  std::vector<Scope *> local_scopes_;
  std::unique_ptr<::ThreadPool> pool_{nullptr};
  std::vector<platform::Place> places_;
  std::vector<std::unique_ptr<ir::Graph>> graphs_;

  std::vector<std::unique_ptr<details::FastThreadedSSAGraphExecutor>>
      executors_;
  ExceptionHolder exception_holder_;

  bool support_partial_feed_{false};
  std::vector<FeedStatus> feed_status_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
