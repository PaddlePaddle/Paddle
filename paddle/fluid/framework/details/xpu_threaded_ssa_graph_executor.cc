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
#include "paddle/fluid/framework/details/xpu_threaded_ssa_graph_executor.h"
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
#include "paddle/fluid/platform/profiler.h"

#if defined(PADDLE_WITH_XPU)
namespace paddle {
namespace framework {
namespace details {

static std::atomic<unsigned int> exec_op_count_;
static std::atomic<int> error_state;

XPUThreadedSSAGraphExecutor::XPUThreadedSSAGraphExecutor(
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
  multi_stream_num_ = 1;
  if (std::getenv("XPU_PADDLE_MULTI_STREAM_NUM") != nullptr) {
    sscanf(std::getenv("XPU_PADDLE_MULTI_STREAM_NUM"), "%d",
           &multi_stream_num_);
  }
  PADDLE_ENFORCE(multi_stream_num_ >= 1, "%d less than 1", multi_stream_num_);
  stream_op_count_.reset(new int[multi_stream_num_ * places.size()]);
  PADDLE_ENFORCE(stream_op_count_.get() != nullptr, "no enough memory\n");
  for (uint32_t i = 0; i < places.size() * multi_stream_num_; i++) {
    pool_.emplace_back(std::unique_ptr<::ThreadPool>(new ::ThreadPool(1)));
    stream_op_count_[i] = 0;
  }
  printf("multi_stream_num: %d place_num:%lu mode:%s\n", multi_stream_num_,
         places.size(), std::getenv("XPU_PADDLE_MULTI_STREAM") ? "MULTI_STREM"
                                                               : "MAIN_STREAM");
  int index = 0;
  for (uint32_t i = 0; i < places.size(); i++) {
    // int id = boost::get<platform::XPUPlace>(places[i]).device;
    int id = BOOST_GET_CONST(platform::XPUPlace, places_[i]).device;
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
  PADDLE_ENFORCE_GT(op_deps_.size(), 0, "The graph doesn't have operators.");
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

int XPUThreadedSSAGraphExecutor::get_pool_thread_index(int device_id) {
  int cur_index = multi_stream_num_ * place_to_index_[device_id];
  int min_num = stream_op_count_[cur_index];
  int index = 0;
  for (int i = 1; i < multi_stream_num_; i++) {
    if (min_num > stream_op_count_[cur_index + i]) {
      index = i;
      min_num = stream_op_count_[cur_index + i];
    }
  }
  return cur_index + index;
}

FetchResultType XPUThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  VLOG(3) << "enter XPUThreadedSSAGraphExecutor Run";
  return RunMainStream(fetch_tensors, return_merged);
}

// use 2 streams to run op. The first stream is main stream and will run
// most op exclude op depending on multi device(e.g., all_reduce, fetch op)
FetchResultType XPUThreadedSSAGraphExecutor::RunMainStream(
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
                 &fetch_ops, &ready_fetch_ops);

  for (auto cur_op : bootstrap_ops_) {
    ready_ops->Push(cur_op);
  }
  for (auto cur_op : ready_fetch_ops) {
    ready_ops->Push(cur_op);
  }

  exec_op_count_ = 0;

  platform::XPUPlace cur_place;
  std::size_t cur_count = 0;
  while (cur_count < op_deps_.size()) {
    cur_count++;
    auto cur_op = ready_ops->Pop();
    if (cur_op == nullptr) {
      // sleep a while to make sure worker thread quit
      sleep(10);
      exec_op_count_ = op_deps_.size();
      break;
    }
    auto dev_ctxes_ = cur_op->DeviceContext();
    if (dev_ctxes_.size() > 1) {
      RunMultiDeviceOpAsync(cur_op, op_deps.get(), ready_ops);
      continue;
    } else if (dev_ctxes_.size() == 1) {
      cur_place = boost::get<platform::XPUPlace>(dev_ctxes_.begin()->first);
    } else {
      cur_place = boost::get<platform::XPUPlace>(
          dynamic_cast<ComputationOpHandle *>(cur_op)->GetPlace());
    }
    int cur_index = multi_stream_num_ * place_to_index_[cur_place.device];
    RunOpAsyncMainStream(cur_op, op_deps.get(), ready_ops, cur_index);
  }
  while (exec_op_count_ < op_deps_.size()) {
  }

  // Wait FetchOps.
  ClearFetchOp(graph_, &fetch_ops);
  if (exception_.IsCaught()) {
    ExecutionFinal(&fetch_ops);
  }
  return fetches;
}

void XPUThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors, FetchResultType *fetches,
    std::unordered_map<std::string, std::vector<VarHandleBase *>> *fetched_vars,
    std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
    std::vector<OpHandleBase *> *fetch_ops,
    std::vector<OpHandleBase *> *ready_fetch_ops) {
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
    PADDLE_ENFORCE(fetched_var_it != fetched_vars->end(),
                   "Cannot find fetched variable(%s).(Perhaps the main_program "
                   "is not set to ParallelExecutor)",
                   var_name);

    auto &vars = fetched_var_it->second;

    ir::Node *fetch_node =
        graph_->CreateEmptyNode("fetch", ir::Node::Type::kOperation);
    auto *op = new FetchOpHandle(fetch_node, fetches, i, &local_scopes_,
                                 &local_exec_scopes_, true);
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

void XPUThreadedSSAGraphExecutor::RunMultiDeviceOpAsync(
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
        op->Run(strategy_.use_cuda_, strategy_.use_xpu_);
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
    exec_op_count_++;
  });
}

void XPUThreadedSSAGraphExecutor::RunOpAsyncMainStream(
    OpHandleBase *op,
    std::unordered_map<OpHandleBase *, struct RunningItem> *op_deps,
    std::shared_ptr<BlockingQueue<OpHandleBase *>> ready_ops, int index) {
  pool_[index]->enqueue([=] {
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    pool.device_context_index = index % multi_stream_num_;
    try {
      if (error_state == 0 && LIKELY(!strategy_.dry_run_)) {
        struct timeval t1;
        struct timeval t2;
        if (std::getenv("XPU_PADDLE_DEBUG_COUNT") != nullptr) {
          gettimeofday(&t1, NULL);
        }
        op->Run(strategy_.use_cuda_, strategy_.use_xpu_);
        auto &outputs = op->Outputs();
        for (auto &output : outputs) {
          for (auto &pending_op : output->PendingOps()) {
            std::atomic<int> &deps = op_deps->at(pending_op).dep_num;
            if (deps.fetch_sub(1) == 1) {
              ready_ops->Push(pending_op);
            }
          }
        }
        if (std::getenv("XPU_PADDLE_DEBUG_COUNT") != nullptr) {
          xpu_wait();
          gettimeofday(&t2, NULL);
          uint32_t diff =
              1000000 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec;
          printf("op [%s] op time: %u\n", op->Name().c_str(), diff);
        }
      } else if (error_state) {
        ready_ops->Push(nullptr);
      }
    } catch (...) {
      error_state = 1;
      ready_ops->Push(nullptr);
      exception_.Catch(std::current_exception());
    }
    exec_op_count_++;
  });
}

void XPUThreadedSSAGraphExecutor::PrepareAtomicOpDeps() {
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

const ir::Graph &XPUThreadedSSAGraphExecutor::Graph() const { return *graph_; }

void XPUThreadedSSAGraphExecutor::ExecutionFinal(
    std::vector<OpHandleBase *> *fetch_ops) {
  VLOG(3) << "caught exception " << exception_.Type() << ", rethrow it";
  ClearFetchOp(graph_, fetch_ops);
  exception_.ReThrow();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
#endif
