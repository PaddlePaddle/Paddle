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
    std::vector<std::unique_ptr<ir::Graph>> &&graphs)
    : strategy_(std::move(strategy)),
      local_scopes_(std::move(local_scopes)),
      pool_(places.size() >= 2 ? new ::ThreadPool(places.size()) : nullptr),
      places_(std::move(places)),
      graphs_(std::move(graphs)) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());

  // set the correct size of thread pool to each device.
  strategy_.num_threads_ = strategy_.num_threads_ < places_.size()
                               ? 1UL
                               : strategy_.num_threads_ / places_.size();
  VLOG(1) << "set num_threads: " << strategy_.num_threads_
          << " to run the operators of the graph on each device.";
  for (size_t i = 0; i < places.size(); ++i) {
    executors_.emplace_back(new details::ThreadedSSAGraphExecutor(
        strategy_, {local_scopes_[i]}, {places_[i]}, std::move(graphs_[i])));
  }
}

FeedFetchList ParallelSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  std::vector<std::future<FeedFetchList>> run_futures;

  std::vector<FeedFetchList> fetch_data;
  FeedFetchList ret;

  fetch_data.reserve(places_.size());
  ret.reserve(fetch_tensors.size());
  exception_holder_.Clear();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto call = [this, i, &fetch_tensors]() -> FeedFetchList {
      try {
        return executors_[i]->Run(fetch_tensors);
      } catch (...) {
        exception_holder_.Catch(std::current_exception());
      }
      return FeedFetchList();
    };

    if (pool_) {
      run_futures.emplace_back(pool_->enqueue(std::move(call)));
    } else {
      fetch_data.emplace_back(call());
    }
  }

  if (pool_) {
    for (auto &f : run_futures) {
      if (exception_holder_.IsCaught()) {
        f.wait();
      } else {
        fetch_data.emplace_back(f.get());
      }
    }
  }
  if (exception_holder_.IsCaught()) {
    exception_holder_.ReThrow();
  }

  for (size_t fetch_idx = 0; fetch_idx < fetch_tensors.size(); ++fetch_idx) {
    std::vector<const LoDTensor *> lodtensor_ptrs;
    lodtensor_ptrs.reserve(local_scopes_.size());
    for (size_t scope_idx = 0; scope_idx < local_scopes_.size(); ++scope_idx) {
      lodtensor_ptrs.push_back(&fetch_data.at(scope_idx).at(fetch_idx));
    }
    ret.emplace_back();
    ret.back().MergeLoDTensor(lodtensor_ptrs, platform::CPUPlace());
  }
  return ret;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
