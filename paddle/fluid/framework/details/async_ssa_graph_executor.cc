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

#include "paddle/fluid/framework/details/async_ssa_graph_executor.h"

#include "paddle/fluid/framework/variable_helper.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/fluid/operators/distributed/communicator.h"
#endif

namespace paddle {
namespace framework {
namespace details {

inline void InitVarsInScope(const std::vector<VarInfo> &var_infos, Scope *scope,
                            Scope *local_scope) {
  VLOG(3) << "InitVarsInScope";
  for (auto &info : var_infos) {
    if (info.persistable_) {  // Persistable
      auto *var = scope->FindVar(info.name_);
      if (var != nullptr) {
        VLOG(2) << info.name_
                << " has been initialized beforehand in global scope, skipped";
        continue;
      }
      InitializeVariable(scope->Var(info.name_), info.type_);
    } else {
      InitializeVariable(local_scope->Var(info.name_), info.type_);
    }
  }
}

// get RpcContext and remote send and recv op
void ProcessGraph(std::vector<ir::Graph *> graphs, Scope *scope) {
#ifdef PADDLE_WITH_DISTRIBUTE
  using RpcCtxMap = operators::distributed::RpcCtxMap;
  VLOG(3) << "ProcessGraph";
  RpcCtxMap send_varname_to_ctx;

  for (auto &node : graphs[0]->Nodes()) {
    VLOG(3) << "node name " << node->Name();
    if (node && node->IsOp()) {
      if (node->Name() == "send") {
        auto send_var_name = node->Op()->Input("X")[0];
        auto send_varnames =
            BOOST_GET_CONST(std::vector<std::string>,
                            node->Op()->GetNullableAttr("send_varnames"));
        auto epmap = BOOST_GET_CONST(std::vector<std::string>,
                                     node->Op()->GetNullableAttr("epmap"));
        auto height_section = BOOST_GET_CONST(
            std::vector<int64_t>, node->Op()->GetNullableAttr("sections"));
        auto trainer_id =
            BOOST_GET_CONST(int, node->Op()->GetNullableAttr("trainer_id"));
        auto merge_add =
            BOOST_GET_CONST(bool, node->Op()->GetNullableAttr("merge_add"));
        if (!merge_add) {
          merge_add = FLAGS_communicator_is_sgd_optimizer;
        }
        auto use_send_handler = BOOST_GET_CONST(
            bool, node->Op()->GetNullableAttr("use_send_handler"));
        send_varname_to_ctx[send_var_name] = operators::distributed::RpcContext(
            send_var_name, send_varnames, epmap, height_section, trainer_id,
            merge_add, use_send_handler);
        VLOG(3) << "find and init an send op: "
                << send_varname_to_ctx[send_var_name];
      }
    }
  }

  // init communicator here
  if (send_varname_to_ctx.size() > 0) {
    auto *instance = operators::distributed::Communicator::GetInstance();
    auto initialized = instance ? true : false;
    PADDLE_ENFORCE_EQ(initialized, true,
                      platform::errors::InvalidArgument(
                          "Communicator is not Initialized, you may use "
                          "FleetAPI(https://github.com/PaddlePaddle/Fleet/tree/"
                          "develop/markdown_doc/transpiler)"));
  }
#endif
}

AsyncSSAGraphExecutor::AsyncSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<Scope *> &local_exec_scopes,
    const std::vector<platform::Place> &places, std::vector<ir::Graph *> graphs)
    : strategy_(std::move(strategy)),
      local_scopes_(std::move(local_scopes)),
      local_exec_scopes_(local_exec_scopes),
      pool_(places.size() >= 2 ? new ::ThreadPool(places.size()) : nullptr),
      places_(std::move(places)),
      graphs_(std::move(graphs)) {
  VLOG(3) << "build AsyncSSAGraphExecutor";
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());
  PADDLE_ENFORCE_EQ(local_scopes_.size(), local_exec_scopes_.size());

  // set the correct size of thread pool to each device.
  strategy_.num_threads_ = strategy_.num_threads_ < places_.size()
                               ? 1UL
                               : strategy_.num_threads_ / places_.size();
  VLOG(1) << "set num_threads: " << strategy_.num_threads_
          << " to run the operators of the graph on each device.";
  for (size_t i = 0; i < places.size(); ++i) {
    executors_.emplace_back(new details::ThreadedSSAGraphExecutor(
        strategy_, {local_scopes_[i]}, {local_exec_scopes_[i]}, {places_[i]},
        graphs_[i]));
  }

  for (auto &node : graphs_[0]->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      var_infos_.emplace_back();
      var_infos_.back().name_ = node->Var()->Name();
      var_infos_.back().type_ = node->Var()->GetType();
      var_infos_.back().persistable_ = node->Var()->Persistable();
    }
  }

  for (size_t i = local_scopes_.size(); i >= 1; --i) {
    InitVarsInScope(var_infos_, local_scopes_[i - 1],
                    local_exec_scopes_[i - 1]);
  }
  ProcessGraph(graphs_, local_scopes_[0]);
}

void AsyncSSAGraphExecutor::StartOffPythonTrainLoop(bool return_merged) {
  VLOG(3) << "StartOffPythonTrainLoop size = " << places_.size();
  for (size_t i = 1; i < places_.size(); ++i) {
    auto call = [this, i, return_merged]() -> void {
      VLOG(3) << "start off python thread " << i;
      try {
        while (true) {
          executors_[i]->Run({}, return_merged);
        }
      } catch (...) {
        exception_holder_.Catch(std::current_exception());
        VLOG(3) << "get exception type = " << exception_holder_.Type();
      }
      VLOG(3) << "thread " << i << " exited!";
    };
    run_futures_.emplace_back(pool_->enqueue(std::move(call)));
  }
}

void AsyncSSAGraphExecutor::HandleException() {
  if (exception_holder_.IsCaught()) {
    for (auto &f : run_futures_) {
      VLOG(3) << "wait future";
      f.wait();
    }
    VLOG(3) << "caught exception " << exception_holder_.Type()
            << ", rethrow it";
    run_futures_.clear();
    exception_holder_.ReThrow();
  }
}

FetchResultType AsyncSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  PADDLE_ENFORCE_EQ(return_merged, true,
                    platform::errors::InvalidArgument(
                        "AsyncSSAGraphExecutor does not support unmerged "
                        "results to be fetched!"));
  // init once
  if (run_futures_.size() == 0 && places_.size() > 1) {
    if (strategy_.thread_barrier_) {
#ifdef PADDLE_WITH_DISTRIBUTE
      operators::distributed::Communicator::GetInstance()->BarrierTriggerReset(
          places_.size());
#endif
    }
    exception_holder_.Clear();
    StartOffPythonTrainLoop(return_merged);
  }

  if (places_.size() == 1) {
    exception_holder_.Clear();
  }

  FetchResultType fetch_data;

  try {
    fetch_data = executors_[0]->Run(fetch_tensors, return_merged);
  } catch (...) {
    exception_holder_.Catch(std::current_exception());
  }

  HandleException();

  FetchList ret;
  auto &val = BOOST_GET(FetchList, fetch_data);
  for (size_t fetch_idx = 0; fetch_idx < fetch_tensors.size(); ++fetch_idx) {
    if (data_is_lod_tensor(val.at(fetch_idx))) {
      std::vector<const LoDTensor *> lodtensor_ptrs;
      lodtensor_ptrs.push_back(&(BOOST_GET(LoDTensor, val.at(fetch_idx))));
      LoDTensor var;
      var.MergeLoDTensor(lodtensor_ptrs, platform::CPUPlace());
      ret.emplace_back(var);
    } else {
      auto array = BOOST_GET(LoDTensorArray, val.at(fetch_idx));
      LoDTensorArray item_array;
      item_array.reserve(array.size());
      for (size_t i = 0; i < array.size(); ++i) {
        std::vector<const LoDTensor *> lodtensor_ptrs;
        lodtensor_ptrs.push_back(&array[i]);
        item_array.emplace_back();
        item_array.back().MergeLoDTensor(lodtensor_ptrs, platform::CPUPlace());
      }
      ret.emplace_back(item_array);
    }
  }
  return ret;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
