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

#include "paddle/fluid/operators/distributed/rpc_server.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace paddle {
namespace operators {
namespace distributed {

void RPCServer::ShutDown() {
  LOG(INFO) << "RPCServer ShutDown ";
  ShutDownImpl();

  exit_flag_ = true;
  // notify barriers to make rpc threads exit.
  send_barrier_->Notify();
  recv_barrier_->Notify();
  state_cond_.notify_all();
}

void RPCServer::SavePort() const {
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  std::ofstream port_file;
  port_file.open(file_path);
  port_file << selected_port_;
  port_file.close();
  VLOG(3) << "selected port written to " << file_path;
}

void RPCServer::Complete() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    num_clients_--;
    need_reset_all_vars_ = true;

    VLOG(4) << "decrease client_num to: " << num_clients_;
    // TODO(typhoonzero): comment why need to decrease here.
    if (state_ == RPCServerState::STATE_SEND) {
      send_barrier_->Decrease();
    } else if (state_ == RPCServerState::STATE_RECV) {
      recv_barrier_->Decrease();
    }
  }
  send_barrier_->SetWorkerSize(num_clients_);
  recv_barrier_->SetWorkerSize(num_clients_);
}

int RPCServer::GetNumClients() {
  std::unique_lock<std::mutex> lock(mutex_);
  return num_clients_;
}

void RPCServer::RegisterRPC(const RequestType req_type, RequestHandler* handler,
                            int thread_num) {
  if (rpc_call_map_.find(req_type) == rpc_call_map_.end()) {
    rpc_thread_num_[req_type] = thread_num;
    rpc_call_map_[req_type] = handler;
  } else {
    LOG(WARNING) << "RPC call handler already registered for type: "
                 << req_type;
  }
}

void RPCServer::SetState(const RPCServerState state) {
  VLOG(3) << "RPCServer SetState " << state;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    state_ = state;
  }
  state_cond_.notify_all();
}

void RPCServer::WaitState(const RPCServerState state) {
  VLOG(4) << "RPCServer WaitCond " << state;
  std::unique_lock<std::mutex> lock(mutex_);
  state_cond_.wait(lock,
                   [=] { return (state_ == state || exit_flag_.load()); });
}

// ----------------------------------------------------------------
// variable scope barriers
void RPCServer::MarkVarReady(const std::string& varname) {
  std::unique_lock<std::mutex> lock(var_ready_mutex_);
  if (var_ready_map_.find(varname) != var_ready_map_.end()) {
    var_ready_map_[varname].reset(new Barrier(num_clients_));
  }
}

void RPCServer::UnmarkVarReady(const std::string& varname) {
  std::unique_lock<std::mutex> lock(var_ready_mutex_);
  if (var_ready_map_.find(varname) != var_ready_map_.end()) {
    var_ready_map_.erase(varname);
  }
}

void RPCServer::WaitVarReady(const std::string& varname) {
  std::unique_lock<std::mutex> lock(var_ready_mutex_);
  if (var_ready_map_.find(varname) != var_ready_map_.end()) {
    var_ready_map_[varname]->Wait();
  }
}

void RPCServer::ResetVarReady() {
  std::unique_lock<std::mutex> lock(var_ready_mutex_);
  if (!var_ready_map_.empty()) {
    var_ready_map_.clear();
  }
}

Barrier* RPCServer::VarReadyBarrier(const std::string& varname) {
  std::unique_lock<std::mutex> lock(var_ready_mutex_);
  if (var_ready_map_.find(varname) != var_ready_map_.end()) {
    return var_ready_map_[varname].get();
  }
  return nullptr;
}

bool RPCServer::NeedResetAllVars() {
  std::unique_lock<std::mutex> lock(mutex_);
  return need_reset_all_vars_;
}

void RPCServer::ResetAllBarriers() {
  send_barrier_->Reset();
  recv_barrier_->Reset();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    need_reset_all_vars_ = false;
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
