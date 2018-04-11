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
    std::unique_ptr<SSAGraph> &&graph, bool allow_op_delay)
    : SSAGraphExecutor(std::move(graph)),
      pool_(num_threads >= 2 ? new ::ThreadPool(num_threads) : nullptr),
      local_scopes_(local_scopes),
      places_(places),
      fetch_ctxs_(places),
      use_event_(use_event),
      running_ops_(0),
      allow_op_delay_(allow_op_delay) {}

void ThreadedSSAGraphExecutor::RunDelayedOps(
    const std::unordered_set<OpHandleBase *> &delayed_ops) {
  for (auto op : delayed_ops) {
    op->Run(use_event_);
  }
}

FeedFetchList ThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  auto run_ctx = PrepareOrGetContext(fetch_tensors);
  BlockingQueue<VarHandleBase *> ready_vars;
  ready_vars.Extend(run_ctx.ready_vars_);
  // For ops (e.g. nccl_all_reduce) that need to coordinate multiple
  // streams from multiple GPUs, it's faster to buffer them and schedule
  // together since we currently cannot overlap computation and memcpy
  // streams. Should revisit it if overlapping is available.
  std::unordered_set<OpHandleBase *> delayed_ops;
  std::unordered_set<OpHandleBase *> blocked_by_delayed_ops;
  std::unordered_set<VarHandleBase *> delayed_vars;

  auto run_all_ready_ops = [&] {
    for (auto *op : run_ctx.ready_ops_) {
      if (op->IsMultiDeviceTransfer() && allow_op_delay_) {
        delayed_ops.insert(op);
        delayed_vars.insert(op->outputs_.begin(), op->outputs_.end());
        ready_vars.Extend(op->outputs_);
        continue;
      }
      running_ops_++;
      RunOp(&ready_vars, op);
    }
    run_ctx.ready_ops_.clear();
  };

  // Create local scopes.
  for (auto &scope : local_scopes_) {
    auto &local_scope = scope->NewScope();
    *scope->Var("@TMP_SCOPE@")->GetMutable<Scope *>() = &local_scope;
  }

  // Step 3. Execution
  while (!run_ctx.pending_vars_.empty() || !run_ctx.ready_ops_.empty() ||
         !delayed_ops.empty()) {
    // 1. Run All Ready ops
    run_all_ready_ops();

    // 2. Find ready variable
    bool timeout;
    auto cur_ready_vars = ready_vars.PopAll(1000, &timeout);

    if (timeout) {
      if (exception_) {
        throw * exception_;
      } else {
        VLOG(10) << "ParallelExecutor is timeout "
                 << run_ctx.pending_vars_.size();
        auto *first_pending_var = (*run_ctx.pending_vars_.begin());
        VLOG(10) << first_pending_var->generated_op_->DebugString();

        continue;
      }
    }
    // 3. Remove the dependency of ready_var.
    // Find the ready_ops after the ready_var.
    for (auto ready_var : cur_ready_vars) {
      run_ctx.pending_vars_.erase(ready_var);
      for (auto *op : ready_var->pending_ops_) {
        auto &deps = run_ctx.pending_ops_[op];
        --deps;
        if (deps == 0) {
          if (delayed_vars.find(ready_var) != delayed_vars.end()) {
            blocked_by_delayed_ops.insert(op);
          } else {
            run_ctx.ready_ops_.insert(op);
          }
        }
      }
    }
    // When there are no other ops to schedule, schedule buffered delayed
    // ops and unblock other ops.
    if (run_ctx.ready_ops_.empty() && !delayed_ops.empty() &&
        running_ops_ == 0) {
      RunDelayedOps(delayed_ops);
      delayed_ops.clear();
      for (auto *op : blocked_by_delayed_ops) {
        run_ctx.ready_ops_.insert(op);
      }
      blocked_by_delayed_ops.clear();
    }
    // Keep loop until all vars are ready.
  }
  PADDLE_ENFORCE(run_ctx.ready_ops_.empty());
  PADDLE_ENFORCE(delayed_ops.empty());
  PADDLE_ENFORCE(blocked_by_delayed_ops.empty());

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
  if (!fetch_tensors.empty()) {
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

  return run_ctx.FetchedResult();
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

ThreadedSSAGraphExecutor::RunContext
ThreadedSSAGraphExecutor::PrepareOrGetContext(
    const std::vector<std::string> &fetch_tensors) {
  if (contexts_.find(fetch_tensors) ==
      contexts_.end()) {  // Cannot find local context. Prepare it.
    auto &context = contexts_[fetch_tensors];

    for (auto &var : graph_->vars_) {
      if (var->generated_op_ != nullptr) {
        context.pending_vars_.emplace(var.get());
      } else {
        context.ready_vars_.push_back(var.get());
      }
    }

    for (auto &op : graph_->ops_) {
      if (op->inputs_.empty()) {
        VLOG(10) << "Insert empty " << op->inputs_.size();
        context.ready_ops_.emplace(op.get());
      } else {
        context.pending_ops_.emplace(op.get(), op->inputs_.size());
      }
    }

    context.fetched_tensors_.resize(fetch_tensors.size());

    std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

    for (auto &fetch_var_name : fetch_tensors) {
      auto &result = fetched_vars[fetch_var_name];
      size_t cur_version = 0;
      for (auto &var : graph_->vars_) {
        auto *v = dynamic_cast<VarHandle *>(var.get());
        if (v == nullptr) {
          continue;
        }

        if (v->name_ != fetch_var_name) {
          continue;
        }

        if (v->version_ < cur_version) {
          continue;
        } else if (v->version_ > cur_version) {
          result.clear();
        }  // Deps on largest version of variable
        result.push_back(v);
      }
    }

    for (size_t i = 0; i < fetch_tensors.size(); ++i) {
      auto &var_name = fetch_tensors[i];
      auto &vars = fetched_vars.at(var_name);
      auto *op =
          new FetchOpHandle(&context.fetched_tensors_, i, &local_scopes_);
      context.fetch_ops_.emplace_back(op);

      for (auto &p : places_) {
        op->dev_ctxes_[p] = fetch_ctxs_.Get(p);
      }

      for (auto *var : vars) {
        op->inputs_.emplace_back(var);
      }

      auto *fetch_dummy = new DummyVarHandle();
      op->AddOutput(fetch_dummy);
      context.fetch_dependencies_.emplace_back(fetch_dummy);
      context.pending_ops_.emplace(op, op->outputs_.size());
      context.pending_vars_.emplace(fetch_dummy);
    }
  }

  return ThreadedSSAGraphExecutor::RunContext(&contexts_[fetch_tensors]);
}

ThreadedSSAGraphExecutor::RunContext::RunContext(
    ThreadedSSAGraphExecutor::StoredContext *stored_context)
    : stored_context_(stored_context) {
  pending_vars_ = stored_context->pending_vars_;
  pending_ops_ = stored_context->pending_ops_;
  ready_ops_ = stored_context->ready_ops_;
  ready_vars_ = stored_context->ready_vars_;

  for (auto &fetch_op : stored_context->fetch_ops_) {
    VLOG(10) << "Insert Input " << &fetch_op;
    fetch_op->InsertInputs();
  }
}

ThreadedSSAGraphExecutor::RunContext::~RunContext() {
  for (auto &fetch_op : stored_context_->fetch_ops_) {
    VLOG(10) << "Erase Input " << &fetch_op;
    fetch_op->RemoveInputs();
  }

  for (auto &tensor : stored_context_->fetched_tensors_) {
    tensor.Clear();
  }
}

std::vector<LoDTensor> ThreadedSSAGraphExecutor::RunContext::FetchedResult()
    const {
  return this->stored_context_->fetched_tensors_;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
