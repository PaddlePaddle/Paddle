// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/details/bind_threaded_ssa_graph_executor.h"
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#if defined(PADDLE_WITH_XPU)
namespace paddle {
namespace framework {
namespace details {

BindThreadedSSAGraphExecutor::BindThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<Scope *> &local_exec_scopes,
    const std::vector<platform::Place> &places, ir::Graph *graph)
    : strategy_(strategy),
      local_scopes_(local_scopes),
      local_exec_scopes_(local_exec_scopes),
      places_(places),
      graph_(graph),
      prepare_pool_(1),
      multi_device_op_pool_(1) {
  for (uint32_t i = 0; i < places.size(); i++) {
    pool_.emplace_back(std::unique_ptr<::ThreadPool>(new ::ThreadPool(1)));
  }
  int index = 0;
  for (uint32_t i = 0; i < places.size(); i++) {
    int id = places_[i].device;
    if (place_to_index_.find(id) == place_to_index_.end()) {
      place_to_index_[id] = index;
      index++;
    }
  }
  for (auto &op : ir::FilterByNodeWrapper<OpHandleBase>(*graph_)) {
    int dep = static_cast<int>(op->NotReadyInputSize());
    op_deps_.emplace(op, dep);
    if (dep == 0) {
      bootstrap_ops_.emplace_back(op);
    }
  }
  PADDLE_ENFORCE_GT(op_deps_.size(), 0,
                    platform::errors::PreconditionNotMet(
                        "The graph doesn't have operators."));
  PrepareAtomicOpDeps();
}

static std::vector<OpHandleBase *> get_children(OpHandleBase *op) {
  auto &outputs = op->Outputs();
  std::vector<OpHandleBase *> ret;
  for (auto &output : outputs) {
    ret.insert(ret.end(), output->PendingOps().begin(),
               output->PendingOps().end());
  }
  return ret;
}

static std::vector<OpHandleBase *> get_parents(OpHandleBase *op) {
  auto &inputs = op->Inputs();
  std::vector<OpHandleBase *> ret;
  for (auto &input : inputs) {
    if (input->GeneratedOp() != nullptr) {
      ret.push_back(input->GeneratedOp());
    }
  }
  return ret;
}

FetchResultType BindThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  VLOG(3) << "enter BindThreadedSSAGraphExecutor Run";
  return RunMainStream(fetch_tensors, return_merged);
}

// use 2 streams to run op. The first stream is main stream and will run
// most op exclude op depending on multi device(e.g., all_reduce, fetch op)
FetchResultType BindThreadedSSAGraphExecutor::RunMainStream(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  VLOG(3) << "enter MainStream Run";
  std::unique_ptr<std::unordered_map<OpHandleBase *, struct RunningItem>>
      op_deps = atomic_op_deps_.get();
  PrepareAtomicOpDeps();

  error_state = 0;
  paddle::framework::FetchResultType fetches;
  if (return_merged) {
    fetches = FetchList(fetch_tensors.size());
  } else {
    fetches = FetchUnmergedList(fetch_tensors.size());
  }
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;
  std::vector<OpHandleBase *> fetch_ops;
  std::vector<OpHandleBase *> ready_fetch_ops;
  auto ready_ops = std::make_shared<BlockingQueue<OpHandleBase *>>();
  exception_.Clear();

  InsertFetchOps(fetch_tensors, &fetches, &fetched_vars, op_deps.get(),
                 &fetch_ops, &ready_fetch_ops, return_merged);
  for (auto cur_op : bootstrap_ops_) {
    ready_ops->Push(cur_op);
  }
  for (auto cur_op : ready_fetch_ops) {
    ready_ops->Push(cur_op);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    exec_op_count_ = 0;
  }

  platform::XPUPlace cur_place;
  std::size_t cur_count = 0;

  while (cur_count < op_deps->size()) {
    cur_count++;
    auto cur_op = ready_ops->Pop();
    // when execption, get cur_op == nullptr
    if (cur_op == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      exec_op_count_ = op_deps->size();
      break;
    }
    auto dev_ctxes_ = cur_op->DeviceContext();
    if (cur_op->IsMultiDeviceTransfer()) {
      RunMultiDeviceOpAsync(cur_op, op_deps.get(), ready_ops);
      continue;
    } else {
      cur_place = dev_ctxes_.begin()->first;
      int cur_index = place_to_index_[cur_place.device];
      RunOpAsyncMainStream(cur_op, op_deps.get(), ready_ops, cur_index);
    }
  }
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return exec_op_count_ >= op_deps->size(); });
  }

  if (exception_.IsCaught()) {
    ExecutionFinal(&fetch_ops);
  }

  // Wait FetchOps.
  ClearFetchOp(graph_, &fetch_ops);
  return fetches;
}

void BindThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors, FetchResultType *fetches,
    std::unordered_map<std::string, std::vector<VarHandleBase *>> *fetched_vars,
    std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
    std::vector<OpHandleBase *> *fetch_ops,
    std::vector<OpHandleBase *> *ready_fetch_ops, bool return_merged) {
  std::unordered_set<std::string> fetch_tensor_set(fetch_tensors.begin(),
                                                   fetch_tensors.end());
  for (auto &fetch_var_name : fetch_tensor_set) {
    for (auto &var_map : graph_->Get<GraphVars>(kGraphVars)) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        (*fetched_vars)[fetch_var_name].push_back(*it->second.rbegin());
      }
    }
  }

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors.at(i);
    auto fetched_var_it = fetched_vars->find(var_name);
    PADDLE_ENFORCE_NE(
        fetched_var_it, fetched_vars->end(),
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
    auto *op = new FetchOpHandle(fetch_node, fetches, i, &local_scopes_,
                                 &local_exec_scopes_, return_merged);
    fetch_ops->emplace_back(op);

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    for (auto &p : places_) {
      op->SetDeviceContext(p, pool.Get(p));
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    int dep = static_cast<int>(op->NotReadyInputSize());
    (*op_deps)[op].dep_num = dep;
    (*op_deps)[op].op = op;
    if (dep == 0) {
      ready_fetch_ops->emplace_back(op);
    }
  }
}
// RunMultiDeviceOpAsync function is used for Communicated OPs
// like all_reduce\broadcast among multicards.
void BindThreadedSSAGraphExecutor::RunMultiDeviceOpAsync(
    OpHandleBase *op,
    std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
    std::shared_ptr<BlockingQueue<OpHandleBase *>> ready_ops) {
  multi_device_op_pool_.enqueue([=] {
    try {
      if (error_state == 0 && LIKELY(!strategy_.dry_run_)) {
        auto dev_ctxes = op->DeviceContext();
        auto &inputs = op->Inputs();
        for (auto &input : inputs) {
          auto dev_ctxes = input->GeneratedOp()->DeviceContext();
          for (auto &item : dev_ctxes) {
            ((platform::XPUDeviceContext *)(item.second))->Wait();
          }
        }
        op->Run(strategy_.use_device_);
        auto &outputs = op->Outputs();
        for (auto &output : outputs) {
          for (auto &pending_op : output->PendingOps()) {
            std::atomic<int> &deps = op_deps->at(pending_op).dep_num;
            if (deps.fetch_sub(1) == 1) {
              ready_ops->Push(pending_op);
            }
          }
        }
      } else if (error_state) {
        ready_ops->Push(nullptr);
      }
    } catch (...) {
      error_state = 1;
      ready_ops->Push(nullptr);
      exception_.Catch(std::current_exception());
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      exec_op_count_++;
      cv_.notify_all();
    }
  });
}
// RunOpAsyncMainStream function is used for computed OPs
void BindThreadedSSAGraphExecutor::RunOpAsyncMainStream(
    OpHandleBase *op,
    std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
    std::shared_ptr<BlockingQueue<OpHandleBase *>> ready_ops, int index) {
  pool_[index]->enqueue([=] {
    try {
      if (error_state == 0 && LIKELY(!strategy_.dry_run_)) {
        op->Run(strategy_.use_device_);
        auto &outputs = op->Outputs();
        for (auto &output : outputs) {
          for (auto &pending_op : output->PendingOps()) {
            std::atomic<int> &deps = op_deps->at(pending_op).dep_num;
            if (deps.fetch_sub(1) == 1) {
              ready_ops->Push(pending_op);
            }
          }
        }
      } else if (error_state) {
        ready_ops->Push(nullptr);
      }
    } catch (...) {
      error_state = 1;
      ready_ops->Push(nullptr);
      exception_.Catch(std::current_exception());
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      exec_op_count_++;
      cv_.notify_all();
    }
  });
}

void BindThreadedSSAGraphExecutor::PrepareAtomicOpDeps() {
  atomic_op_deps_ = prepare_pool_.enqueue([&] {
    auto *op_deps = new std::unordered_map<OpHandleBase *, struct RunningItem>;
    for (auto &pair : op_deps_) {
      (*op_deps)[pair.first].dep_num = pair.second;
      (*op_deps)[pair.first].op = pair.first;
    }
    return std::unique_ptr<
        std::unordered_map<OpHandleBase *, struct RunningItem>>(op_deps);
  });
}

const ir::Graph &BindThreadedSSAGraphExecutor::Graph() const { return *graph_; }

void BindThreadedSSAGraphExecutor::ExecutionFinal(
    std::vector<OpHandleBase *> *fetch_ops) {
  VLOG(3) << "caught exception " << exception_.Type() << ", rethrow it";
  ClearFetchOp(graph_, fetch_ops);
  exception_.ReThrow();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
#endif
