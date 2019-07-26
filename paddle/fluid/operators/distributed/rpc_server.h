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

#pragma once

#include <atomic>
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/operators/distributed/barrier.h"
#include "paddle/fluid/operators/distributed/request.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace paddle {
namespace operators {
namespace distributed {

enum RPCServerState { STATE_SEND, STATE_RECV, STATE_NONE };

class RPCServer {
 public:
  explicit RPCServer(const std::string& address, int num_clients)
      : state_(RPCServerState::STATE_RECV),
        send_barrier_(new Barrier(num_clients)),
        recv_barrier_(new Barrier(num_clients)),
        bind_address_(address),
        exit_flag_(false),
        selected_port_(0),
        num_clients_(num_clients),
        need_reset_all_vars_(false) {}

  virtual ~RPCServer() {}

  // ----------------------------------------------------------------
  // Interfaces that implementations should have:
  virtual void StartServer() = 0;
  virtual void WaitServerReady() = 0;
  // ----------------------------------------------------------------

  void ShutDown();

  bool IsExit() { return exit_flag_.load(); }

  int GetSelectedPort() const { return selected_port_; }

  int GetNumClients();

  void SavePort() const;

  void Complete();

  int GetThreadNum(const RequestType req_type) {
    return rpc_thread_num_[req_type];
  }

  // ----------------------------------------------------------------
  // RegisterRPC, register the rpc method name to a handler
  // class, and auto generate a condition id for this call
  // to be used for the barrier.
  void RegisterRPC(const RequestType req_type, RequestHandler* handler,
                   int thread_num = 5);

  // ----------------------------------------------------------------
  // For sync training, server side barrier controls
  void SetState(const RPCServerState state);
  void WaitState(const RPCServerState state);
  Barrier* SendBarrier() { return send_barrier_.get(); }
  Barrier* RecvBarrier() { return recv_barrier_.get(); }
  void ResetAllBarriers();

  // TODO(typhoonzero): in here or in handler or in collective server?
  // mark variable ready for workers to fetch, and only for fetch n
  // (num_workers) times then the barrier will be removed.
  // TODO(typhoonzero): Should use var ready barriers for recv
  void MarkVarReady(const std::string& varname);
  void UnmarkVarReady(const std::string& varname);
  void WaitVarReady(const std::string& varname);
  Barrier* VarReadyBarrier(const std::string& varname);
  void ResetVarReady();
  // ----------------------------------------------------------------
  bool NeedResetAllVars();

 protected:
  virtual void ShutDownImpl() = 0;

 private:
  std::mutex mutex_;
  RPCServerState state_;
  std::condition_variable state_cond_;

  std::unique_ptr<Barrier> send_barrier_;
  std::unique_ptr<Barrier> recv_barrier_;

  std::mutex var_ready_mutex_;
  std::unordered_map<std::string, std::unique_ptr<Barrier>> var_ready_map_;

 protected:
  std::string bind_address_;
  std::atomic<int> exit_flag_;
  int selected_port_;
  int num_clients_;
  bool need_reset_all_vars_;

  std::unordered_map<RequestType, RequestHandler*, EnumClassHash> rpc_call_map_;
  std::unordered_map<RequestType, int, EnumClassHash> rpc_thread_num_;
  friend class RequestHandler;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
