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

inline void NewTempScopeAndInitVars(const std::vector<VarInfo> &var_infos,
                                    Scope *scope) {
  VLOG(3) << "NewTempScopeAndInitVars";
  Scope &local_scope = scope->NewScope();
  *scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>() =
      &local_scope;

  for (auto &info : var_infos) {
    if (scope->FindVar(info.name_) != nullptr) {
      continue;
    }

    if (info.persistable_) {  // Persistable
      InitializeVariable(scope->Var(info.name_), info.type_);
    } else {
      InitializeVariable(local_scope.Var(info.name_), info.type_);
    }
  }
}

// get RpcContext and remote send and recv op
void ProcessGraph(std::vector<ir::Graph *> graphs, Scope *scope) {
#ifdef PADDLE_WITH_DISTRIBUTE
  using RpcCtxMap = operators::distributed::RpcCtxMap;
  VLOG(3) << "ProcessGraph";
  RpcCtxMap send_varname_to_ctx;
  RpcCtxMap recv_varname_to_ctx;
  for (auto i = 0; i < graphs.size(); ++i) {
    std::vector<ir::Node *> nodes_to_delete;
    for (auto &node : graphs[i]->Nodes()) {
      VLOG(3) << "node name " << node->Name();
      if (node && node->IsOp()) {
        if (node->Name() == "send") {
          auto send_var_name = node->Op()->Input("X")[0];
          auto send_varnames = boost::get<std::vector<std::string>>(
              node->Op()->GetNullableAttr("send_varnames"));
          auto epmap = boost::get<std::vector<std::string>>(
              node->Op()->GetNullableAttr("epmap"));
          auto height_section = boost::get<std::vector<int64_t>>(
              node->Op()->GetNullableAttr("sections"));
          send_varname_to_ctx[send_var_name] =
              operators::distributed::RpcContext(send_var_name, send_varnames,
                                                 epmap, height_section);
          VLOG(3) << "find and init an send op: "
                  << send_varname_to_ctx[send_var_name];
        } else if (node->Name() == "recv") {
          auto recv_var_name = node->Op()->Output("Out")[0];
          auto recv_varnames = boost::get<std::vector<std::string>>(
              node->Op()->GetNullableAttr("recv_varnames"));
          auto epmap = boost::get<std::vector<std::string>>(
              node->Op()->GetNullableAttr("epmap"));
          recv_varname_to_ctx[recv_var_name] =
              operators::distributed::RpcContext(recv_var_name, recv_varnames,
                                                 epmap, {});
          nodes_to_delete.push_back(node);
          VLOG(3) << "find and remove an recv op: "
                  << recv_varname_to_ctx[recv_var_name];
        } else if (node->Name() == "lookup_table") {
          VLOG(0) << "set lookup_table op remote_prefetch to false";
          node->Op()->SetAttr("remote_prefetch", false);
        }
      }
    }
  }
  // init communicator here
  if (send_varname_to_ctx.size() > 0) {
    VLOG(3) << "this is distribute mode, will use communicator";
    operators::distributed::Communicator::Init(send_varname_to_ctx,
                                               recv_varname_to_ctx, scope);
    operators::distributed::Communicator::GetInstance()->Start();
  }
#endif
}

AsyncSSAGraphExecutor::AsyncSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, std::vector<ir::Graph *> graphs)
    : strategy_(std::move(strategy)),
      local_scopes_(std::move(local_scopes)),
      pool_(places.size() >= 2 ? new ::ThreadPool(places.size()) : nullptr),
      places_(std::move(places)),
      graphs_(std::move(graphs)) {
  VLOG(3) << "build AsyncSSAGraphExecutor";
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());

  // set the correct size of thread pool to each device.
  strategy_.num_threads_ = strategy_.num_threads_ < places_.size()
                               ? 1UL
                               : strategy_.num_threads_ / places_.size();
  VLOG(1) << "set num_threads: " << strategy_.num_threads_
          << " to run the operators of the graph on each device.";
  for (size_t i = 0; i < places.size(); ++i) {
    executors_.emplace_back(new details::ThreadedSSAGraphExecutor(
        strategy_, {local_scopes_[i]}, {places_[i]}, graphs_[i]));
  }

  for (auto &node : graphs_[0]->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      var_infos_.emplace_back();
      var_infos_.back().name_ = node->Var()->Name();
      var_infos_.back().type_ = node->Var()->GetType();
      var_infos_.back().persistable_ = node->Var()->Persistable();
    }
  }
  for (auto *scope : local_scopes_) {
    NewTempScopeAndInitVars(var_infos_, scope);
  }
  ProcessGraph(graphs_, local_scopes_[0]);
}

void AsyncSSAGraphExecutor::StartOffPythonTrainLoop() {
  VLOG(3) << "StartOffPythonTrainLoop size = " << places_.size();
  for (size_t i = 1; i < places_.size(); ++i) {
    auto call = [this, i]() -> void {
      VLOG(3) << "start off python thread " << i;
      try {
        while (true) {
          executors_[i]->Run({});
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

FeedFetchList AsyncSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  // init once
  if (run_futures_.size() == 0 && places_.size() > 1) {
    exception_holder_.Clear();
    StartOffPythonTrainLoop();
  }

  if (places_.size() == 1) {
    exception_holder_.Clear();
  } else {
    HandleException();
  }

  FeedFetchList fetch_data;
  fetch_data.reserve(fetch_tensors.size());

  try {
    fetch_data = executors_[0]->Run(fetch_tensors);
  } catch (...) {
    exception_holder_.Catch(std::current_exception());
  }

  HandleException();

  FeedFetchList ret;
  for (size_t fetch_idx = 0; fetch_idx < fetch_tensors.size(); ++fetch_idx) {
    std::vector<const LoDTensor *> lodtensor_ptrs;
    lodtensor_ptrs.push_back(&fetch_data.at(fetch_idx));
    ret.emplace_back();
    ret.back().MergeLoDTensor(lodtensor_ptrs, platform::CPUPlace());
  }
  return ret;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
