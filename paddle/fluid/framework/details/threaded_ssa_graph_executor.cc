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

#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {
ThreadedSSAGraphExecutor::ThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, ir::Graph *graph)
    : graph_(graph),
      pool_(strategy.num_threads_ >= 2 ? new ::ThreadPool(strategy.num_threads_)
                                       : nullptr),
      prepare_pool_(1),
      local_scopes_(local_scopes),
      places_(places),
      fetch_ctxs_(places),
      strategy_(strategy),
      running_ops_(0) {
  PrepareOpDeps();
  CopyOpDeps();
}

FeedFetchList ThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  std::unique_ptr<platform::RecordEvent> event(
      new platform::RecordEvent("ThreadedSSAGraphExecutorPrepare"));
  std::unique_ptr<OpDependentData> op_deps = op_deps_futures_.get();
  CopyOpDeps();
  VLOG(10) << "ThreadedSSAGraphExecutor::Run";
  std::shared_ptr<BlockingQueue<VarHandleBase *>> ready_vars(
      new BlockingQueue<VarHandleBase *>);
  auto &pending_ops = op_deps->pending_ops_;
  auto &pending_vars = op_deps->pending_vars_;
  auto &ready_ops = op_deps->ready_ops_;

  // For ops (e.g. nccl_all_reduce) that need to coordinate multiple
  // streams from multiple GPUs, it's faster to buffer them and schedule
  // together since we currently cannot overlap computation and memcpy streams.
  // Should revisit it if overlapping is available.
  std::unordered_set<OpHandleBase *> delayed_ops;

  // Step 2. Insert FetchOps
  std::vector<FetchOpHandle *> fetch_ops;
  std::unordered_set<VarHandleBase *> fetch_dependencies;
  FeedFetchList fetch_data(fetch_tensors.size());

  InsertFetchOps(fetch_tensors, &fetch_ops, &fetch_dependencies, &ready_ops,
                 &pending_ops, &pending_vars, &fetch_data);

  auto run_all_ops = [&](std::unordered_set<OpHandleBase *> &set) {
    for (auto *op : set) {
      running_ops_++;
      RunOp(ready_vars, op);
    }
    set.clear();
  };
  // Clean run context
  run_op_futures_.clear();
  exception_holder_.Clear();
  event.reset(nullptr);
  // Step 3. Execution
  while (!pending_vars.empty()) {
    // 1. Run All Ready ops
    // Keep loop until all vars are ready.
    // NOTE: DelayedOps have a lower priority. It will be scheduled after all
    // ready_ops have been performed.
    if (ready_ops.empty() && strategy_.allow_op_delay_ && running_ops_ == 0) {
      run_all_ops(delayed_ops);
    } else {
      run_all_ops(ready_ops);
    }

    // 2. Find ready variable
    bool timeout;
    auto cur_ready_vars = ready_vars->PopAll(1, &timeout);
    if (timeout) {
      if (exception_holder_.IsCaught()) {
        for (auto &run_op_future : run_op_futures_) {
          run_op_future.wait();
        }
        ClearFetchOp(graph_, &fetch_ops);
        exception_holder_.ReThrow();
      } else {
        continue;
      }
    }

    // 3. Remove the dependency of ready_var.
    // Find the ready_ops after the ready_var.
    for (auto ready_var : cur_ready_vars) {
      pending_vars.erase(ready_var);
      for (auto *op : ready_var->PendingOps()) {
        auto &deps = pending_ops[op];
        --deps;
        if (deps == 0) {
          if (op->IsMultiDeviceTransfer() && strategy_.allow_op_delay_) {
            delayed_ops.insert(op);
          } else {
            ready_ops.insert(op);
          }
        }
      }
    }
  }
  PADDLE_ENFORCE(ready_ops.empty());
  // Wait FetchOps.
  ClearFetchOp(graph_, &fetch_ops);

  return fetch_data;
}

void ThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors,
    std::vector<FetchOpHandle *> *fetch_ops,
    std::unordered_set<VarHandleBase *> *fetch_dependencies,
    std::unordered_set<OpHandleBase *> *ready_ops,
    std::unordered_map<OpHandleBase *, size_t> *pending_ops,
    std::unordered_set<VarHandleBase *> *pending_vars,
    FeedFetchList *fetch_data) {
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;
  std::unordered_set<VarHandleBase *> local_ready_vars;
  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : graph_->Get<details::GraphVars>(details::kGraphVars)) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].emplace_back(*it->second.rbegin());
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
    auto *op = new FetchOpHandle(fetch_node, fetch_data, i, &local_scopes_);
    fetch_ops->emplace_back(op);

    for (auto &p : places_) {
      op->SetDeviceContext(p, fetch_ctxs_.Get(p));
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    ir::Node *fetch_var =
        graph_->CreateEmptyNode("fetch", ir::Node::Type::kVariable);
    auto *fetch_dummy = new DummyVarHandle(fetch_var);
    op->AddOutput(fetch_dummy);
    fetch_dependencies->emplace(fetch_dummy);

    this->InsertPendingVar(pending_vars, &local_ready_vars, fetch_dummy);

    size_t wait_input_num = 0;
    std::unordered_set<VarHandleBase *> input_set(vars.begin(), vars.end());
    for (auto *var : input_set) {
      if (pending_vars->count(var)) {
        ++wait_input_num;
      }
    }
    if (wait_input_num) {
      pending_ops->insert({op, wait_input_num});
    } else {
      ready_ops->insert(static_cast<OpHandleBase *>(op));
    }
  }
  PADDLE_ENFORCE_EQ(local_ready_vars.size(), 0);
}

void ThreadedSSAGraphExecutor::InsertPendingOp(
    std::unordered_map<OpHandleBase *, size_t> *pending_ops,
    OpHandleBase *op_instance) const {
  pending_ops->insert({op_instance, op_instance->NoDupInputSize()});
}

void ThreadedSSAGraphExecutor::InsertPendingVar(
    std::unordered_set<VarHandleBase *> *pending_vars,
    std::unordered_set<VarHandleBase *> *ready_vars, VarHandleBase *var) const {
  pending_vars->insert(var);
  if (var->GeneratedOp() == nullptr) {
    ready_vars->insert(var);
  }
}

void ThreadedSSAGraphExecutor::PrepareOpDeps() {
  op_deps_.reset(new OpDependentData());
  std::unordered_map<OpHandleBase *, size_t> &pending_ops =
      op_deps_->pending_ops_;
  std::unordered_set<VarHandleBase *> &pending_vars = op_deps_->pending_vars_;
  std::unordered_set<OpHandleBase *> &ready_ops = op_deps_->ready_ops_;
  std::unordered_set<VarHandleBase *> ready_vars;

  // Transform SSAGraph to pending_ops & pending_vars
  for (auto &var_map : graph_->Get<details::GraphVars>(details::kGraphVars)) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        InsertPendingVar(&pending_vars, &ready_vars, version_pair);
      }
    }
  }
  for (auto &var : graph_->Get<details::GraphDepVars>(details::kGraphDepVars)) {
    InsertPendingVar(&pending_vars, &ready_vars, var);
  }

  for (auto &op : ir::FilterByNodeWrapper<OpHandleBase>(*graph_)) {
    if (op->Inputs().empty()) {  // Special case, Op has no input.
      ready_ops.insert(op);
    } else {
      InsertPendingOp(&pending_ops, op);
    }
  }
  for (auto ready_var : ready_vars) {
    pending_vars.erase(ready_var);
    for (auto *op : ready_var->PendingOps()) {
      auto &deps = pending_ops[op];
      --deps;
      if (deps == 0) {
        ready_ops.insert(op);
      }
    }
  }
}

void ThreadedSSAGraphExecutor::CopyOpDeps() {
  op_deps_futures_ = prepare_pool_.enqueue([&] {
    auto *op_deps = new OpDependentData();
    op_deps->pending_ops_.insert(op_deps_->pending_ops_.begin(),
                                 op_deps_->pending_ops_.end());
    op_deps->pending_vars_.insert(op_deps_->pending_vars_.begin(),
                                  op_deps_->pending_vars_.end());
    op_deps->ready_ops_.insert(op_deps_->ready_ops_.begin(),
                               op_deps_->ready_ops_.end());
    return std::unique_ptr<OpDependentData>(op_deps);
  });
}

void ThreadedSSAGraphExecutor::RunOp(
    const std::shared_ptr<BlockingQueue<VarHandleBase *>> &ready_var_q,
    details::OpHandleBase *op) {
  auto op_run = [ready_var_q, op, this] {
    try {
      if (VLOG_IS_ON(10)) {
        VLOG(10) << op << " " << op->Name() << " : " << op->DebugString();
      }
      if (LIKELY(!strategy_.dry_run_)) {
        op->Run(strategy_.use_cuda_);
      }
      VLOG(10) << op << " " << op->Name() << " Done ";
      running_ops_--;
      ready_var_q->Extend(op->Outputs());
      VLOG(10) << op << " " << op->Name() << " Signal posted";
    } catch (...) {
      exception_holder_.Catch(std::current_exception());
    }
  };
  if (pool_) {
    run_op_futures_.emplace_back(pool_->enqueue(op_run));
  } else {
    op_run();
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
