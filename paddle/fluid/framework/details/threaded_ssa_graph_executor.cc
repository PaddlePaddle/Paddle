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

namespace paddle {
namespace framework {
namespace details {
ThreadedSSAGraphExecutor::ThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    std::unique_ptr<SSAGraph> &&graph)
    : graph_(std::move(graph)),
      pool_(strategy.num_threads_ >= 2 ? new ::ThreadPool(strategy.num_threads_)
                                       : nullptr),
      local_scopes_(local_scopes),
      places_(places),
      fetch_ctxs_(places),
      running_ops_(0),
      strategy_(strategy) {}

FeedFetchList ThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  std::unordered_map<OpHandleBase *, size_t> pending_ops;
  std::unordered_set<VarHandleBase *> pending_vars;
  BlockingQueue<VarHandleBase *> ready_vars;
  std::unordered_set<OpHandleBase *> ready_ops;
  // For ops (e.g. nccl_all_reduce) that need to coordinate multiple
  // streams from multiple GPUs, it's faster to buffer them and schedule
  // together since we currently cannot overlap computation and memcpy streams.
  // Should revisit it if overlapping is available.
  std::unordered_set<OpHandleBase *> delayed_ops;

  // Transform SSAGraph to pending_ops & pending_vars
  for (auto &var_map : graph_->vars_) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        InsertPendingVar(&pending_vars, &ready_vars, version_pair.get());
      }
    }
  }
  for (auto &var : graph_->dep_vars_) {
    InsertPendingVar(&pending_vars, &ready_vars, var.get());
  }

  for (auto &op : graph_->ops_) {
    if (op->Inputs().empty()) {  // Special case, Op has no input.
      ready_ops.insert(op.get());
    } else {
      InsertPendingOp(&pending_ops, op.get());
    }
  }

  // Step 2. Insert FetchOps
  std::vector<std::unique_ptr<FetchOpHandle>> fetch_ops;
  std::unordered_set<std::unique_ptr<VarHandleBase>> fetch_dependencies;
  FeedFetchList fetch_data(fetch_tensors.size());

  InsertFetchOps(fetch_tensors, &fetch_ops, &fetch_dependencies, &pending_ops,
                 &pending_vars, &ready_vars, &fetch_data);

  auto run_all_ops = [&](std::unordered_set<OpHandleBase *> &set) {
    for (auto *op : set) {
      running_ops_++;
      RunOp(&ready_vars, op);
    }
    set.clear();
  };

  // Step 3. Execution
  while (!pending_vars.empty()) {
    // 1. Run All Ready ops
    // Keep loop until all vars are ready.
    //
    // NOTE: DelayedOps have a lower priority. It will be scheduled after all
    // ready_ops have been performed.
    if (ready_ops.empty() && strategy_.allow_op_delay_ && running_ops_ == 0) {
      run_all_ops(delayed_ops);
    } else {
      run_all_ops(ready_ops);
    }

    // 2. Find ready variable
    bool timeout;
    auto cur_ready_vars = ready_vars.PopAll(1, &timeout);

    if (timeout) {
      std::lock_guard<std::mutex> l(exception_mu_);
      if (exception_) {
        auto exp = *exception_;
        exception_.reset();
        throw exp;
      } else {
        continue;
      }
    }
    // 3. Remove the dependency of ready_var.
    // Find the ready_ops after the ready_var.
    for (auto ready_var : cur_ready_vars) {
      pending_vars.erase(ready_var);
      for (auto *op : ready_var->pending_ops_) {
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
  if (!fetch_ops.empty()) {
    fetch_ops.clear();
  }

  return fetch_data;
}

void ThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors,
    std::vector<std::unique_ptr<FetchOpHandle>> *fetch_ops,
    std::unordered_set<std::unique_ptr<VarHandleBase>> *fetch_dependencies,
    std::unordered_map<OpHandleBase *, size_t> *pending_ops,
    std::unordered_set<VarHandleBase *> *pending_vars,
    BlockingQueue<VarHandleBase *> *ready_vars, FeedFetchList *fetch_data) {
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : graph_->vars_) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(it->second.rbegin()->get());
      }
    }
  }

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto &vars = fetched_vars.at(var_name);
    auto *op = new FetchOpHandle(fetch_data, i, &local_scopes_);
    fetch_ops->emplace_back(op);

    for (auto &p : places_) {
      op->SetDeviceContext(p, fetch_ctxs_.Get(p));
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    auto *fetch_dummy = new DummyVarHandle();
    op->AddOutput(fetch_dummy);
    fetch_dependencies->emplace(fetch_dummy);
    this->InsertPendingVar(pending_vars, ready_vars, fetch_dummy);
    this->InsertPendingOp(pending_ops, op);
  }
}

void ThreadedSSAGraphExecutor::InsertPendingOp(
    std::unordered_map<OpHandleBase *, size_t> *pending_ops,
    OpHandleBase *op_instance) const {
  pending_ops->insert({op_instance, op_instance->NoDupInputSize()});
}

void ThreadedSSAGraphExecutor::InsertPendingVar(
    std::unordered_set<VarHandleBase *> *pending_vars,
    BlockingQueue<VarHandleBase *> *ready_vars, VarHandleBase *var) const {
  pending_vars->insert(var);
  if (var->generated_op_ == nullptr) {
    ready_vars->Push(var);
  }
}

void ThreadedSSAGraphExecutor::RunOp(
    BlockingQueue<VarHandleBase *> *ready_var_q, details::OpHandleBase *op) {
  auto op_run = [ready_var_q, op, this] {
    try {
      if (VLOG_IS_ON(10)) {
        VLOG(10) << op << " " << op->Name() << " : " << op->DebugString();
      }
      op->Run(strategy_.use_cuda_);
      VLOG(10) << op << " " << op->Name() << " Done ";
      running_ops_--;
      ready_var_q->Extend(op->Outputs());
      VLOG(10) << op << " " << op->Name() << "Signal posted";
    } catch (platform::EnforceNotMet ex) {
      std::lock_guard<std::mutex> l(exception_mu_);
      exception_.reset(new platform::EnforceNotMet(ex));
    } catch (...) {
      LOG(FATAL) << "Unknown exception catched";
    }
  };
  if (pool_) {
    pool_->enqueue(op_run);
  } else {
    op_run();
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
