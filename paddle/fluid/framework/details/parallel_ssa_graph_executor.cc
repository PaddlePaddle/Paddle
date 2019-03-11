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
#include <memory>
#include <utility>
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

std::vector<std::unique_ptr<ir::Graph>>
ParallelSSAGraphExecutor::SeparateMultiDevicesGraph(ir::Graph *graph) {
  std::vector<std::unique_ptr<ir::Graph>> graphs;
  graphs.reserve(places_.size());
  for (size_t i = 0; i < places_.size(); ++i) {
    ProgramDesc empty;
    graphs.emplace_back(std::unique_ptr<ir::Graph>(new ir::Graph(empty)));
    auto &g = graphs.back();
    g->Set(kGraphVars, new GraphVars(1UL));
    g->Set(kGraphDepVars, new GraphDepVars);
    auto &stale_ops =
        graph->Get<const std::vector<OpDesc *>>(details::kStaleProgramOpDescs);
    g->Erase(details::kStaleProgramOpDescs);
    g->Set<const std::vector<OpDesc *>>(details::kStaleProgramOpDescs,
                                        new std::vector<OpDesc *>(stale_ops));
  }
  auto op_handles = ir::FilterByNodeWrapper<OpHandleBase>(*graph);

  for (auto &op : op_handles) {
    auto &dev_ctx = op->DeviceContext();
    auto &p = dev_ctx.begin()->first;
    int dev_id = boost::get<platform::CUDAPlace>(p).device;
    auto &dev_dummys = graphs[dev_id]->Get<GraphDepVars>(kGraphDepVars);
    graphs[dev_id]->AddNode(graph->RemoveNode(op->Node()).release());

    for (auto &var : op->Inputs()) {
      auto dummy_ptr = dynamic_cast<DummyVarHandle *>(var);
      if (dummy_ptr) {
        dev_dummys.insert(var);
        if (graph->Nodes().count(var->Node()))
          graphs[dev_id]->AddNode(graph->RemoveNode(var->Node()).release());
      }
    }
    for (auto &var : op->Outputs()) {
      auto dummy_ptr = dynamic_cast<DummyVarHandle *>(var);
      if (dummy_ptr) {
        dev_dummys.insert(var);
        if (graph->Nodes().count(var->Node()))
          graphs[dev_id]->AddNode(graph->RemoveNode(var->Node()).release());
      }
    }
  }

  for (size_t dev_id = 0; dev_id < places_.size(); ++dev_id) {
    auto &dev_vars = graphs[dev_id]->Get<GraphVars>(kGraphVars)[0];
    auto &origin_vars = graph->Get<GraphVars>(kGraphVars)[dev_id];
    for (auto &name_pair : origin_vars) {
      dev_vars.emplace(name_pair.first, name_pair.second);
      for (auto &version_pair : name_pair.second) {
        if (graph->Nodes().count(version_pair->Node())) {
          graphs[dev_id]->AddNode(
              graph->RemoveNode(version_pair->Node()).release());
        }
      }
    }
  }

  return graphs;
}

ParallelSSAGraphExecutor::ParallelSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, ir::Graph *graph)
    : strategy_(std::move(strategy)),
      local_scopes_(std::move(local_scopes)),
      pool_(places.size() >= 2 ? new ::ThreadPool(places.size()) : nullptr),
      places_(std::move(places)),
      // TODO(Yancey1989): Copying graphs is not safely since it deleted the
      // attrs.
      graphs_(SeparateMultiDevicesGraph(graph)) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());

  auto seq_allreduce_pass =
      ir::PassRegistry::Instance().Get("all_reduce_deps_pass");
  for (size_t i = 0; i < graphs_.size(); ++i) {
    graphs_[i] = seq_allreduce_pass->Apply(std::move(graphs_[i]));
  }

  // set the correct size of thread pool to each device.
  strategy_.num_threads_ = strategy_.num_threads_ < places_.size()
                               ? 1UL
                               : strategy_.num_threads_ / places_.size();
  VLOG(1) << "set num_threads: " << strategy_.num_threads_
          << " to run the operators of the graph on each device.";
  for (size_t i = 0; i < places.size(); ++i) {
    executors_.emplace_back(new details::ThreadedSSAGraphExecutor(
        strategy_, local_scopes_, {places_[i]}, graphs_.at(i).get()));
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
