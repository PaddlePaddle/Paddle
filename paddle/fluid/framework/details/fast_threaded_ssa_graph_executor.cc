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
#include "paddle/fluid/framework/details/fast_threaded_ssa_graph_executor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

FastThreadedSSAGraphExecutor::FastThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, ir::Graph *graph)
    : strategy_(strategy),
      local_scopes_(local_scopes),
      places_(places),
      graph_(graph),
      pool_(strategy.num_threads_),
      prepare_pool_(1),  // add one more thread for generate op_deps
      fetch_ctxs_(places) {
  for (auto &op : ir::FilterByNodeWrapper<OpHandleBase>(*graph_)) {
    int dep = static_cast<int>(op->NotReadyInputSize());
    op_deps_.emplace(op, dep);
    if (dep == 0) {
      bootstrap_ops_.emplace_back(op);
    }
  }

  PrepareAtomicOpDeps();
}

FeedFetchList FastThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  std::unique_ptr<std::unordered_map<OpHandleBase *, std::atomic<int>>>
      op_deps = atomic_op_deps_.get();
  PrepareAtomicOpDeps();

  paddle::framework::FeedFetchList fetches;
  fetches.resize(fetch_tensors.size());
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;
  std::vector<FetchOpHandle *> fetch_ops;
  std::vector<OpHandleBase *> ready_fetch_ops;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : graph_->Get<details::GraphVars>(details::kGraphVars)) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(*it->second.rbegin());
      }
    }
  }

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto fetched_var_it = fetched_vars.find(var_name);
    PADDLE_ENFORCE(fetched_var_it != fetched_vars.end(),
                   "Cannot find fetched variable(%s).(Perhaps the main_program "
                   "is not set to ParallelExecutor)",
                   var_name);

    auto &vars = fetched_var_it->second;

    ir::Node *fetch_node =
        graph_->CreateEmptyNode("fetch", ir::Node::Type::kOperation);
    auto *op = new FetchOpHandle(fetch_node, &fetches, i, &local_scopes_);
    fetch_ops.emplace_back(op);

    for (auto &p : places_) {
      op->SetDeviceContext(p, fetch_ctxs_.Get(p));
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    int dep = static_cast<int>(op->NotReadyInputSize());
    (*op_deps)[op] = dep;
    if (dep == 0) {
      ready_fetch_ops.emplace_back(op);
    }
  }

  size_t num_complete = 0;
  remaining_ = 0;
  auto complete_q = std::make_shared<BlockingQueue<size_t>>();
  for (auto op : bootstrap_ops_) {
    RunOpAsync(op_deps.get(), op, complete_q);
  }
  for (auto op : ready_fetch_ops) {
    RunOpAsync(op_deps.get(), op, complete_q);
  }
  while (num_complete != op_deps->size()) {
    size_t num_comp = complete_q->Pop();
    if (num_comp == -1UL) {
      int remaining = 0;
      while (true) {
        remaining = remaining_;
        if (remaining == 0) {
          break;
        }
        for (int i = 0; i < remaining; ++i) {
          complete_q->Pop();
        }
      }
      if (exception_.IsCaught()) {
        ClearFetchOp(graph_, &fetch_ops);
        exception_.ReThrow();
      }
    }
    num_complete += num_comp;
  }
  // Wait FetchOps.
  ClearFetchOp(graph_, &fetch_ops);
  return fetches;
}

void FastThreadedSSAGraphExecutor::RunOpAsync(
    std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
    OpHandleBase *op,
    const std::shared_ptr<BlockingQueue<size_t>> &complete_q) {
  ++remaining_;
  this->pool_.enqueue([=] {
    OpHandleBase *op_to_run = op;
    size_t complete = 0;
    while (op_to_run != nullptr) {
      try {
        if (LIKELY(!strategy_.dry_run_)) {
          op_to_run->Run(strategy_.use_cuda_);
        }
        ++complete;
      } catch (...) {
        exception_.Catch(std::current_exception());
        --remaining_;
        complete_q->Push(-1UL);
        return;
      }
      auto &outputs = op_to_run->Outputs();
      op_to_run = nullptr;
      for (auto &output : outputs) {
        for (auto &pending_op : output->PendingOps()) {
          std::atomic<int> &deps = op_deps->at(pending_op);
          if (deps.fetch_sub(1) == 1) {  // pending_op ready
            if (op_to_run == nullptr) {
              op_to_run = pending_op;
            } else {
              RunOpAsync(op_deps, pending_op, complete_q);
            }
          }
        }
      }
    }
    --remaining_;
    complete_q->Push(complete);
  });
}
void FastThreadedSSAGraphExecutor::PrepareAtomicOpDeps() {
  atomic_op_deps_ = prepare_pool_.enqueue([&] {
    auto *op_deps = new std::unordered_map<OpHandleBase *, std::atomic<int>>;
    for (auto &pair : op_deps_) {
      (*op_deps)[pair.first] = pair.second;
    }
    return std::unique_ptr<
        std::unordered_map<OpHandleBase *, std::atomic<int>>>(op_deps);
  });
}

const ir::Graph &FastThreadedSSAGraphExecutor::Graph() const { return *graph_; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
