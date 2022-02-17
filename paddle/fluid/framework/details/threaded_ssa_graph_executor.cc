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
#include "paddle/fluid/platform/profiler/event_tracing.h"

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#endif

namespace paddle {
namespace framework {
namespace details {
ThreadedSSAGraphExecutor::ThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<Scope *> &local_exec_scopes,
    const std::vector<platform::Place> &places, ir::Graph *graph)
    : graph_(graph),
      local_scopes_(local_scopes),
      local_exec_scopes_(local_exec_scopes),
      places_(places),
      fetch_ctxs_(places),
      strategy_(strategy),
      prepare_pool_(1),
      pool_(strategy.num_threads_ >= 2 ? new ::ThreadPool(strategy.num_threads_)
                                       : nullptr) {
  if (strategy_.num_iteration_per_run_ > 1) {
    int read_op_num = 0;
    for (auto *node : graph_->Nodes()) {
      if (node->IsOp() && node->Name() == "read") {
        read_op_num++;
      }
    }
    if (read_op_num == 0) {
      LOG(WARNING) << "when num_iteration_per_run_ is larger then 1, the model "
                      "should use pyreader to feed data!";
    }
  }
  PrepareOpDeps();
  CopyOpDeps();
}

inline FetchResultType ThreadedSSAGraphExecutor::RunImpl(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  std::unique_ptr<platform::RecordEvent> event(
      new platform::RecordEvent("ThreadedSSAGraphExecutorPrepare",
                                platform::TracerEventType::UserDefined, 2));
  std::unique_ptr<OpDependentData> op_deps = op_deps_futures_.get();
  CopyOpDeps();

  VLOG(10) << "ThreadedSSAGraphExecutor::Run";
  std::shared_ptr<BlockingQueue<VarHandleBase *>> ready_vars(
      new BlockingQueue<VarHandleBase *>);
  auto &pending_ops = op_deps->pending_ops_;
  auto &pending_vars = op_deps->pending_vars_;
  auto &ready_ops = op_deps->ready_ops_;
  size_t num_ops = op_deps->num_ops_;

  // Step 2. Insert FetchOps
  std::vector<OpHandleBase *> fetch_ops;
  std::unordered_set<VarHandleBase *> fetch_dependencies;
  FetchResultType fetch_data;
  if (return_merged) {
    fetch_data = FetchList(fetch_tensors.size());
  } else {
    fetch_data = FetchUnmergedList(fetch_tensors.size());
  }

  InsertFetchOps(fetch_tensors, &fetch_ops, &fetch_dependencies, &ready_ops,
                 &pending_ops, &pending_vars, &fetch_data, return_merged);

  exception_holder_.Clear();
  event.reset(nullptr);

  // Step 3. Execution
  if (strategy_.num_threads_ == 1 && traced_ops_.size() == num_ops) {
    // If the num_threads is 1, we can record the order of operator's
    // execution in the first iteration, and in subsequent iterations,
    // run the recorded operators directly. This strategy could make the
    // execution faster.
    VLOG(3) << "Run the traced ops.";
    bool is_exception_free =
        RunTracedOps(traced_ops_) && RunTracedOps(fetch_ops);
    if (!is_exception_free) {
      ExecutionFinal(&fetch_ops);
    }
  } else {
    traced_ops_.clear();
    auto run_all_ops = [&](std::unordered_set<OpHandleBase *> &set) {
      for (auto *op : set) {
        RunOp(ready_vars, op);
      }
      set.clear();
    };
    // Clean run context
    run_op_futures_.clear();

    while (!pending_vars.empty()) {
      // 1. Run All Ready ops
      // Keep loop until all vars are ready.
      run_all_ops(ready_ops);

      // 2. Find ready variable
      bool timeout;
      auto cur_ready_vars = ready_vars->PopAll(1, &timeout);
      if (timeout) {
        for (auto &run_op_future : run_op_futures_) {
          run_op_future.wait();
        }
        if (exception_holder_.IsCaught()) {
          ExecutionFinal(&fetch_ops);
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
            ready_ops.insert(op);
          }
        }
      }
    }
    PADDLE_ENFORCE_EQ(
        ready_ops.empty(), true,
        platform::errors::Fatal("After the execution of computation graph, "
                                "there are unexecuted operators left."));
  }

  // Wait FetchOps.
  ClearFetchOp(graph_, &fetch_ops);

  return fetch_data;
}

FetchResultType ThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  for (size_t j = 0; j < strategy_.num_iteration_per_run_ - 1; ++j) {
    RunImpl({}, return_merged);
  }
  return RunImpl(fetch_tensors, return_merged);
}

void ThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors,
    std::vector<OpHandleBase *> *fetch_ops,
    std::unordered_set<VarHandleBase *> *fetch_dependencies,
    std::unordered_set<OpHandleBase *> *ready_ops,
    std::unordered_map<OpHandleBase *, size_t> *pending_ops,
    std::unordered_set<VarHandleBase *> *pending_vars,
    FetchResultType *fetch_data, bool return_merged) {
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
    PADDLE_ENFORCE_NE(
        fetched_var_it, fetched_vars.end(),
        platform::errors::PreconditionNotMet(
            "Cannot find fetched variable(%s) in current computation graph. "
            "Possible reasons are:\n"
            "  1. The variable to be fetched is not defined in main program.\n"
            "  2. The variable to be fetched is not an input or output of any "
            "operator.\n"
            "  3. Confirm that you have used the fetch `Variable` format "
            "instead of the string literal('%s') in `fetch_list` parameter "
            "when using `executor.run` method. In other words, the format of "
            "`executor.run(fetch_list=[fetch_var])`(fetch_var is a Variable) "
            "is recommended.",
            var_name, var_name));

    auto &vars = fetched_var_it->second;

    ir::Node *fetch_node =
        graph_->CreateEmptyNode("fetch", ir::Node::Type::kOperation);
    auto *op = new FetchOpHandle(fetch_node, fetch_data, i, &local_scopes_,
                                 &local_exec_scopes_, return_merged);
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
  PADDLE_ENFORCE_EQ(
      local_ready_vars.size(), 0,
      platform::errors::Fatal(
          "The number of ready variables should be 0, but got %d.",
          local_ready_vars.size()));
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
  op_deps_->num_ops_ = ready_ops.size() + pending_ops.size();
  PADDLE_ENFORCE_GT(
      op_deps_->num_ops_, 0,
      platform::errors::InvalidArgument("The graph doesn't have operators."));

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
    op_deps->num_ops_ = op_deps_->num_ops_;
    return std::unique_ptr<OpDependentData>(op_deps);
  });
}

void ThreadedSSAGraphExecutor::RunOp(
    const std::shared_ptr<BlockingQueue<VarHandleBase *>> &ready_var_q,
    details::OpHandleBase *op) {
  auto op_run = [ready_var_q, op, this] {
    RunOpSync(op);
    try {
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

  RecordOps(op);
}

bool ThreadedSSAGraphExecutor::RunTracedOps(
    const std::vector<OpHandleBase *> &traced_ops) {
  for (auto &op : traced_ops) {
    if (!RunOpSync(op)) return false;
  }
  return true;
}

bool ThreadedSSAGraphExecutor::RunOpSync(OpHandleBase *op) {
  try {
    VLOG(10) << op << " " << op->Name() << " : " << op->DebugString();
    if (LIKELY(!strategy_.dry_run_)) {
      op->Run(strategy_.use_device_);
    }
    VLOG(10) << op << " " << op->Name() << " Done ";
    return true;
  } catch (...) {
    exception_holder_.Catch(std::current_exception());
    return false;
  }
}

void ThreadedSSAGraphExecutor::ExecutionFinal(
    std::vector<OpHandleBase *> *fetch_ops) {
#if defined PADDLE_WITH_PSCORE
  if (strategy_.thread_barrier_) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerDecrement();
  }
#endif
  VLOG(3) << "caught exception " << exception_holder_.Type() << ", rethrow it";
  ClearFetchOp(graph_, fetch_ops);
  exception_holder_.ReThrow();
}

void ThreadedSSAGraphExecutor::RecordOps(OpHandleBase *op) {
  if (strategy_.num_threads_ == 1 && !dynamic_cast<FetchOpHandle *>(op)) {
    traced_ops_.emplace_back(op);
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
