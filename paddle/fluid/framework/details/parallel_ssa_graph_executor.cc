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

#include <algorithm>
#include <memory>
#include <utility>

#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

static std::vector<std::unique_ptr<ir::Graph>> SeparateMultiDevicesGraph(
    ir::Graph *graph, size_t place_num) {
  std::vector<std::unique_ptr<ir::Graph>> graphs;
  graphs.reserve(place_num);
  for (size_t i = 0; i < place_num; ++i) {
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
    int dev_id = p.device;
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

  for (size_t dev_id = 0; dev_id < place_num; ++dev_id) {
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
    const std::vector<Scope *> &local_exec_scopes,
    const std::vector<platform::Place> &places, ir::Graph *graph)
    // TODO(Yancey1989): Copying graphs is not safely since it deleted the
    // attrs.
    : ParallelSSAGraphExecutor(strategy, local_scopes, local_exec_scopes,
                               places,
                               SeparateMultiDevicesGraph(graph,
                                                         places.size())) {}

ParallelSSAGraphExecutor::ParallelSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<Scope *> &local_exec_scopes,
    const std::vector<platform::Place> &places,
    std::vector<std::unique_ptr<ir::Graph>> graphs)
    : strategy_(std::move(strategy)),
      local_scopes_(std::move(local_scopes)),
      pool_(places.size() >= 2 ? new ::ThreadPool(places.size()) : nullptr),
      places_(places),
      graphs_(std::move(graphs)),
      feed_status_(places.size(), FeedStatus::kNone) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "The number of places and the number of local scopes "
                        "should be equal, but got number of places is %d and "
                        "number of local scopes is %d.",
                        places_.size(), local_scopes_.size()));

  PADDLE_ENFORCE_EQ(places_.size(), graphs_.size(),
                    platform::errors::InvalidArgument(
                        "Graph number does not match place number"));

  PADDLE_ENFORCE_GT(
      places_.size(), 0,
      platform::errors::InvalidArgument("place number must be larger than 0"));

  auto seq_allreduce_pass =
      ir::PassRegistry::Instance().Get("all_reduce_deps_pass");
  seq_allreduce_pass->Set<bool>(kUseHierarchicalAllReduce, new bool(false));
  for (size_t i = 0; i < graphs_.size(); ++i) {
    graphs_[i].reset(seq_allreduce_pass->Apply(graphs_[i].release()));
  }

  // set the correct size of thread pool to each device.
  strategy_.num_threads_ = strategy_.num_threads_ < places_.size()
                               ? 1UL
                               : strategy_.num_threads_ / places_.size();
  VLOG(1) << "set num_threads: " << strategy_.num_threads_
          << " to run the operators of the graph on each device.";
  for (size_t i = 0; i < places.size(); ++i) {
    executors_.emplace_back(new details::FastThreadedSSAGraphExecutor(
        strategy_, local_scopes_, local_exec_scopes, {places_[i]},
        graphs_.at(i).get()));
  }
}

std::vector<ir::Graph *> ParallelSSAGraphExecutor::Graphs() {
  std::vector<ir::Graph *> result;
  result.reserve(graphs_.size());
  for (auto &g : graphs_) {
    result.emplace_back(g.get());
  }
  return result;
}

enum ExceptionStatus { kSuccess = 0, kEOF, kOther };

FetchResultType ParallelSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  size_t feed_num = std::count(feed_status_.begin(), feed_status_.end(),
                               FeedStatus::kHasFeed);
  bool has_feed = (feed_num > 0);

  VLOG(10) << "Feed num " << feed_num;

  size_t place_num = places_.size();

  std::vector<std::future<FetchResultType>> run_futures;
  std::vector<ExceptionStatus> exception_status(place_num,
                                                ExceptionStatus::kSuccess);

  std::vector<FetchResultType> fetch_data;
  fetch_data.reserve(place_num);
  exception_holder_.Clear();

  for (size_t i = 0; i < place_num; ++i) {
    auto call = [&, i]() -> FetchResultType {
      try {
        if (!support_partial_feed_ || !has_feed ||
            feed_status_[i] == FeedStatus::kHasFeed) {
          return executors_[i]->Run(fetch_tensors, return_merged);
        }
      } catch (platform::EOFException &) {
        exception_status[i] = ExceptionStatus::kEOF;
        exception_holder_.Catch(std::current_exception());
      } catch (...) {
        exception_status[i] = ExceptionStatus::kOther;
        exception_holder_.Catch(std::current_exception());
      }

      if (return_merged) {
        return FetchList();
      } else {
        return FetchUnmergedList();
      }
    };

    if (pool_) {
      run_futures.emplace_back(pool_->enqueue(std::move(call)));
    } else {
      fetch_data.emplace_back(call());
    }
  }

  if (pool_) {
    for (auto &f : run_futures) {
      fetch_data.emplace_back(f.get());
    }
  }

  bool has_exception = exception_holder_.IsCaught();
  if (!support_partial_feed_ && has_exception) {
    VLOG(10) << "Exception rethrow because partial feed is not supported";
    exception_holder_.ReThrow();
  }

  std::vector<bool> is_valid(place_num, true);

  if (support_partial_feed_) {
    if (has_feed) {
      for (size_t i = 0; i < place_num; ++i) {
        if (feed_status_[i] == FeedStatus::kNone) {
          is_valid[i] = false;
        } else if (exception_status[i] != ExceptionStatus::kSuccess) {
          PADDLE_ENFORCE_EQ(has_exception, true,
                            platform::errors::InvalidArgument(
                                "Thread pool raises exception but not caught"));
          VLOG(10) << "Exception rethrow because non-EOF exception raises when "
                      "feed is given";
          exception_holder_.ReThrow();
        }
      }
    } else {
      for (size_t i = 0; i < place_num; ++i) {
        if (exception_status[i] == ExceptionStatus::kOther) {
          PADDLE_ENFORCE_EQ(has_exception, true,
                            platform::errors::InvalidArgument(
                                "Thread pool raises exception but not caught"));
          VLOG(10) << "Exception rethrow because non-EOF exception raises when "
                      "feed is not given";
          exception_holder_.ReThrow();
        } else if (exception_status[i] != ExceptionStatus::kSuccess) {
          is_valid[i] = false;
        }
      }
    }
  }

  if (std::count(is_valid.begin(), is_valid.end(), true) == 0) {
    PADDLE_ENFORCE_EQ(has_exception, true,
                      platform::errors::InvalidArgument(
                          "Thread pool raises exception but not caught"));
    VLOG(10) << "Raise exception because there is no success worker";
    exception_holder_.ReThrow();
  }

  if (return_merged) {
    FetchList ret;
    ret.reserve(fetch_tensors.size());
    for (size_t fetch_idx = 0; fetch_idx < fetch_tensors.size(); ++fetch_idx) {
      std::vector<const LoDTensor *> lodtensor_ptrs;
      lodtensor_ptrs.reserve(place_num);
      std::vector<const LoDTensorArray *> lodtensorarray_ptrs;
      lodtensorarray_ptrs.reserve(place_num);
      for (size_t scope_idx = 0; scope_idx < place_num; ++scope_idx) {
        if (!is_valid[scope_idx]) {
          continue;
        }
        const auto &fetch_list =
            BOOST_GET_CONST(FetchList, fetch_data[scope_idx]);
        if (data_is_lod_tensor(fetch_list[fetch_idx])) {
          lodtensor_ptrs.push_back(
              &(BOOST_GET_CONST(LoDTensor, fetch_list[fetch_idx])));
        } else {
          lodtensorarray_ptrs.push_back(
              &(BOOST_GET_CONST(LoDTensorArray, fetch_list[fetch_idx])));
        }
      }
      if (lodtensor_ptrs.size() != 0) {
        LoDTensor var;
        MergeLoDTensor(&var, lodtensor_ptrs, platform::CPUPlace());
        ret.emplace_back(var);
      } else {
        LoDTensorArray var_array(lodtensorarray_ptrs[0]->size());
        for (size_t i = 0; i < lodtensorarray_ptrs[0]->size(); ++i) {
          LoDTensor var;
          std::vector<const LoDTensor *> ptrs;
          for (size_t j = 0; j < lodtensorarray_ptrs.size(); ++j) {
            ptrs.push_back(&(lodtensorarray_ptrs[j]->at(i)));
          }
          MergeLoDTensor(&var, ptrs, platform::CPUPlace());
          var_array[i] = std::move(var);
        }
        ret.emplace_back(var_array);
      }
    }
    return ret;
  } else {
    FetchUnmergedList ret;
    ret.reserve(fetch_tensors.size());
    for (size_t fetch_idx = 0; fetch_idx < fetch_tensors.size(); ++fetch_idx) {
      ret.emplace_back();
      for (size_t scope_idx = 0; scope_idx < local_scopes_.size();
           ++scope_idx) {
        if (!is_valid[scope_idx]) {
          continue;
        }
        const auto &fetch_list =
            BOOST_GET_CONST(FetchUnmergedList, fetch_data[scope_idx]);
        PADDLE_ENFORCE_EQ(
            fetch_list[fetch_idx].size(), 1,
            platform::errors::Fatal("Each place must have only one fetched "
                                    "LoDTensor/LoDTensorArray!"));
        ret.back().emplace_back(fetch_list[fetch_idx][0]);
      }
    }
    return ret;
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
