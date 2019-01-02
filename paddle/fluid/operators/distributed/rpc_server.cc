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
#include "paddle/fluid/platform/profiler.h"

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
  VLOG(4) << "selected port written to " << file_path;
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
  if (var_ready_map_.find(varname) != var_ready_map_.end()) {
    var_ready_map_[varname].reset(new Barrier(num_clients_));
  }
}

void RPCServer::WaitVarReady(const std::string& varname) {
  if (var_ready_map_.find(varname) != var_ready_map_.end()) {
    var_ready_map_[varname]->Wait();
  }
}

void RPCServer::ResetVarReady() {
  if (!var_ready_map_.empty()) {
    var_ready_map_.clear();
  }
}

// void RPCServer::RegisterVar(const std::string& var_name,
//                             const std::string& rpc_name,
//                             framework::Scope* scope,
//                             platform::DeviceContext* dev_ctx) {
//   MonomerHandle h;
//   h.var_name_ = var_name;
//   h.rpc_name_ = rpc_name;
//   h.scope_ = scope;
//   h.dev_ctx_ = dev_ctx;

//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     if (var_map_.find(var_name) != var_map_.end()) {
//       PADDLE_ENFORCE(false, "%s alreay in var_map", var_name);
//     }
//     var_map_[var_name] = h;
//   }

//   rpc_cond_.notify_all();
//   VLOG(4) << "RegisterVar context:" << h.String();
// }

// void RPCServer::IncreaseVarBarrier(const std::string& var_name) {
//   int b = 0;
//   MonomerHandle h;
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     b = ++var_map_[var_name].barrier_;
//     h = var_map_[var_name];
//   }

//   if (b >= num_clients_) {
//     barrier_cond_.notify_all();
//   }

//   VLOG(4) << "IncreaseVarBarrier context:" << h.String();
// }

// void RPCServer::WaitVarBarrier(const std::string& var_name) {
//   VLOG(4) << "WaitBarrier var_name:" << var_name;

//   std::unique_lock<std::mutex> lock(mutex_);
//   barrier_cond_.wait(lock, [&]() {
//     return (
//         (var_map_[var_name].barrier_ >= num_clients_ && num_clients_ != 0) ||
//         exit_flag_.load());
//   });

//   VLOG(4) << "WaitBarrier context: " << var_map_[var_name].String();
// }

// void RPCServer::SetVarCond(const std::string& var_name) {
//   VLOG(4) << "SetVarCond var_name:" << var_name;
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     if (var_map_.find(var_name) != var_map_.end()) {
//       rpc_cond_.notify_all();
//     }
//   }
// }

// void RPCServer::WaitVarCond(const std::string& var_name) {
//   VLOG(4) << "WaitVarCond var_name:" << var_name;

//   std::unique_lock<std::mutex> lock(mutex_);
//   rpc_cond_.wait(lock, [=] {
//     return (var_map_.find(var_name) != var_map_.end() || exit_flag_.load());
//   });

//   VLOG(4) << "WaitVarCond var_name:" << var_name << " end";
// }

// MonomerHandle RPCServer::GetMonomer(const std::string& var_name) {
//   MonomerHandle h;
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     h = var_map_[var_name];
//   }

//   return h;
// }

// void RPCServer::ClearRegisteredVars() {
//   std::unique_lock<std::mutex> lock(mutex_);
//   var_map_.clear();
// }

// void RPCServer::ClearVar(const std::string& var_name) {
//   std::unique_lock<std::mutex> lock(mutex_);
//   var_map_.erase(var_name);
// }
}  // namespace distributed
}  // namespace operators
}  // namespace paddle
