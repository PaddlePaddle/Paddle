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

#include "paddle/fluid/framework/details/fetch_op_handle.h"

namespace paddle {
namespace framework {
namespace details {
ThreadedSSAGraphExecutor::ThreadedSSAGraphExecutor(
    size_t num_threads, bool use_event,
    const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    std::unique_ptr<SSAGraph> &&graph)
    : SSAGraphExecutor(std::move(graph)),
      pool_(num_threads >= 2 ? new ::ThreadPool(num_threads) : nullptr),
      local_scopes_(local_scopes),
      places_(places),
      fetch_ctxs_(places),
      use_event_(use_event),
      running_ops_(0) {}

void ThreadedSSAGraphExecutor::RunDelayedOps(
    const std::unordered_set<OpHandleBase *> &delayed_ops) {
  for (auto op : delayed_ops) {
    op->Run(use_event_);
  }
}

FeedFetchList ThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  std::unordered_map<OpHandleBase *, size_t> pending_ops;
  std::unordered_set<VarHandleBase *> pending_vars;
  BlockingQueue<VarHandleBase *> ready_vars;
  std::unordered_set<OpHandleBase *> ready_ops;

  std::unordered_set<OpHandleBase *> delayed_ops;
  std::unordered_set<OpHandleBase *> after_delayed_ops;
  std::unordered_set<VarHandleBase *> delayed_vars;

  auto InsertPendingVar = [&pending_vars, &ready_vars](VarHandleBase &var) {
    pending_vars.insert(&var);
    if (var.generated_op_ == nullptr) {
      ready_vars.Push(&var);
    }
  };

  auto InsertPendingOp = [&pending_ops](OpHandleBase &op_instance) {
    pending_ops.insert({&op_instance, op_instance.inputs_.size()});
  };

  // Transform SSAGraph to pending_ops & pending_vars
  for (auto &var_map : graph_->vars_) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        InsertPendingVar(version_pair.second);
      }
    }
  }
  for (auto &var : graph_->dep_vars_) {
    InsertPendingVar(*var);
  }

  for (auto &op : graph_->ops_) {
    if (op->inputs_.empty()) {  // Special case, Op has no input.
      ready_ops.insert(op.get());
    } else {
      InsertPendingOp(*op);
    }
  }

  // Step 2. Insert FetchOps
  std::vector<std::unique_ptr<FetchOpHandle>> fetch_ops;
  std::vector<DummyVarHandle> dummy_vars;
  FeedFetchList fetch_data(fetch_tensors.size());

  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : graph_->vars_) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(&it->second.rbegin()->second);
      }
    }
  }

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto &vars = fetched_vars.at(var_name);
    auto *op = new FetchOpHandle(&fetch_data, i, &local_scopes_);
    fetch_ops.emplace_back(op);

    // FIXME: Use new device context
    for (auto &p : places_) {
      op->dev_ctxes_[p] = fetch_ctxs_.Get(p);
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }
    InsertPendingOp(*op);
  }

  auto run_all_ready_ops = [&] {
    for (auto *op : ready_ops) {
      if (op->IsDelayedOp()) {
        delayed_ops.insert(op);
        delayed_vars.insert(op->outputs_.begin(), op->outputs_.end());
        ready_vars.Extend(op->outputs_);
        continue;
      }
      running_ops_++;
      RunOp(&ready_vars, op);
    }
    ready_ops.clear();
  };

  // Create local scopes.
  for (auto &scope : local_scopes_) {
    auto &local_scope = scope->NewScope();
    *scope->Var("@TMP_SCOPE@")->GetMutable<Scope *>() = &local_scope;
  }

  // Step 3. Execution
  while (!pending_vars.empty()) {
    // 1. Run All Ready ops
    run_all_ready_ops();

    // 2. Find ready variable
    bool timeout;
    auto cur_ready_vars = ready_vars.PopAll(1, &timeout);

    if (timeout) {
      if (exception_) {
        throw * exception_;
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
          if (delayed_vars.find(ready_var) != delayed_vars.end()) {
            after_delayed_ops.insert(op);
          } else {
            ready_ops.insert(op);
          }
        }
      }
    }
    if (ready_ops.empty() && !delayed_ops.empty() && running_ops_ == 0) {
      RunDelayedOps(delayed_ops);
      delayed_ops.clear();
      for (auto *op : after_delayed_ops) {
        ready_ops.insert(op);
      }
      after_delayed_ops.clear();
    }
    // Keep loop until all vars are ready.
  }
  ++computation_count_;

  auto sync_computation = [&] {
    computation_count_ = 0;
    // Wait All computational streams
    for (auto p : this->places_) {
      platform::DeviceContextPool::Instance().Get(p)->Wait();
    }
    for (auto &scope : local_scopes_) {
      scope->DropKids();
    }
  };

  // Wait FetchOps.
  if (!fetch_ops.empty()) {
    fetch_ops.clear();
    sync_computation();
  }

  if (computation_count_ == max_async_computation) {
    sync_computation();
  }

  // NOTE: the temp scope can be dropped lazily if needed.
  // Drop tmp scopes;
  for (auto &scope : local_scopes_) {
    auto &kid = *scope->Var("@TMP_SCOPE@")->GetMutable<Scope *>();
    kid = nullptr;
  }

  return fetch_data;
}

void ThreadedSSAGraphExecutor::RunOp(
    BlockingQueue<VarHandleBase *> *ready_var_q, details::OpHandleBase *op) {
  auto op_run = [ready_var_q, op, this] {
    try {
      VLOG(10) << op->Name() << " : " << op->DebugString();
      op->Run(use_event_);
      running_ops_--;
      ready_var_q->Extend(op->outputs_);
    } catch (platform::EnforceNotMet ex) {
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
