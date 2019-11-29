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
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace distributed {

struct MonomerHandle {
  std::string var_name_;
  std::string rpc_name_;
  framework::Scope* scope_{nullptr};
  platform::DeviceContext* dev_ctx_{nullptr};
  int64_t barrier_{0};

  std::string String() {
    std::stringstream ss;
    ss << "var_name:" << var_name_ << ", rpc_name:" << rpc_name_
       << ", scope:" << scope_ << ", dev_ctx:" << dev_ctx_
       << ", barrier_:" << barrier_;
    return ss.str();
  }
};

class RPCServer {
 public:
  explicit RPCServer(const std::string& address, int client_num)
      : cur_cond_(0),
        bind_address_(address),
        exit_flag_(false),
        selected_port_(0),
        client_num_(client_num),
        need_reset_all_vars_(false) {}

  virtual ~RPCServer() {}
  virtual void StartServer() = 0;
  virtual void WaitServerReady() = 0;

  void ShutDown();

  bool IsExit() { return exit_flag_.load(); }

  int GetSelectedPort() const { return selected_port_; }

  int GetClientNum();

  void SavePort() const;

  // RegisterRPC, register the rpc method name to a handler
  // class, and auto generate a condition id for this call
  // to be used for the barrier.
  void RegisterRPC(const std::string& rpc_name, RequestHandler* handler,
                   int thread_num = 5);

  int GetThreadNum(const std::string& rpc_name) {
    return rpc_thread_num_[rpc_name];
  }

  // Wait util all the clients have reached the barrier for one
  // rpc method. This function should be called in the
  // RequestHandler if you want to run the server/client in a
  // synchronous mode.
  void WaitBarrier(const std::string& rpc_name);

  void SetCond(const std::string& rpc_name);
  void WaitCond(const std::string& rpc_name);
  void IncreaseBatchBarrier(const std::string rpc_name);

  void RegisterVar(const std::string& var_name, const std::string& rpc_name,
                   framework::Scope* scope, platform::DeviceContext* dev_ctx);
  void IncreaseVarBarrier(const std::string& var_name);
  void WaitVarBarrier(const std::string& var_name);
  void SetVarCond(const std::string& var_name);
  void WaitVarCond(const std::string& var_name);
  void ClearRegisteredVars();
  void ClearVar(const std::string& var_name);
  MonomerHandle GetMonomer(const std::string& var_name);

  void Complete();

  void ResetBarrierCounter();

  bool NeedResetAllVars();

 protected:
  virtual void ShutDownImpl() = 0;

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, int> barrier_counter_;
  std::condition_variable barrier_cond_;

  std::unordered_map<std::string, int> rpc_cond_map_;
  std::atomic<int> cur_cond_;
  std::condition_variable rpc_cond_;

 protected:
  std::string bind_address_;
  std::atomic<int> exit_flag_;
  int selected_port_;
  int client_num_;
  bool need_reset_all_vars_;

  std::unordered_map<std::string, RequestHandler*> rpc_call_map_;
  std::unordered_map<std::string, int> rpc_thread_num_;
  friend class RequestHandler;

  // TODO(gongwb): use more cond to notify or wait;
  std::unordered_map<std::string, MonomerHandle> var_map_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
