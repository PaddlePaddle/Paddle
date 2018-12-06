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

#include "paddle/fluid/framework/details/parallel_ssa_graph_executor.h"

namespace paddle {
namespace framework {
namespace details {

ParallelSSAGraphExecutor::ParallelSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    std::vector<std::unique_ptr<ir::Graph>> graphs)
    : strategy_(std::move(strategy)),
      local_scopes_(std::move(local_scopes)),
      places_(std::move(places)),
      graphs_(std::move(graphs)),
      pool_(places.size() >= 2 ? new ::ThreadPool(places.size()) : nullptr) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());
  for (size_t i = 0; i < places.size(); ++i) {
    std::vector<framework::Scope *> scopes = {local_scopes_[i]};
    std::vector<platform::Place> places = {places_[i]};
    executors_.emplace_back(new details::ThreadedSSAGraphExecutor(
        strategy_, scopes, places, std::move(graphs_[i])));
  }
}

FeedFetchList ParallelSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  std::vector<std::future<void>> run_futures;
  FeedFetchList fetch_data;

  for (size_t i = 0; i < places_.size(); ++i) {
    auto call = [this, i] {
      // FIXME(Yancey1989): need to fix fetch data failed.
      std::vector<std::string> empty;
      executors_[i]->Run(empty);
    };
    if (pool_) {
      run_futures.emplace_back(pool_->enqueue(std::move(call)));
    } else {
      call();
    }
  }
  if (pool_) {
    for (auto &f : run_futures) {
      f.wait();
    }
  }
  return fetch_data;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
