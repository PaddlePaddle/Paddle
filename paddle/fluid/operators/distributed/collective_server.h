/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace operators {
namespace distributed {
class GatherGetHandler final : public RequestHandler {
 public:
  explicit RequestGetHandler() : RequestHandler(true) {}
  virtual ~RequestGetHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const std::string& out_var_name = "") override {
    if (varname == FETCH_BARRIER_MESSAGE) {
      VLOG(30) << "sync: recv fetch barrier message";
      rpc_server_->IncreaseBatchBarrier(kRequestGet);
    } else {
      rpc_server_->WaitCond(kRequestGet);
      *outvar = scope_->FindVar(varname);
    }
  }

 private:
};

class CollectiveSever final {
 public:
  explicit CollectiveSever(const std::string& end_point, int fan_in);

  virtual ~CollectiveSever() {}

  void StartServer();
  // 1. SetNotReady
  // 2. ResetContext
  void ResetContext(framework::Scope* scope,
                    framework::DeviceContext* dev_ctx) {
    // std::unique_lock<std::mutex> lock(mutex_ready_);
    get_handler_->SetScope(scope);
    get_handler_->SetDevCtx(dev_ctx);
  }

  static CollectiveSever* GetInstance(const std::string& end_point,
                                      int fan_in) {
    std::call_once(init_flag_, [&]() {
      if (collective_server_.get() == nullptr) {
        collective_server_.reset(new CollectiveSever(end_point, fan_in));
        collective_server_->StartServer();
      }
    });

    return collective_server_.get();
  }
  void SetReady() { SetStatus(true); }

  void WaitReady() {
    VLOG(40) << "CollectiveServer WaitReady ";
    std::unique_lock<std::mutex> lock(mutex_ready_);
    rpc_cond_.wait(lock, [=] { return ready_; });
  }

  void WaitNotReady() {
    VLOG(40) << "CollectiveServer WaitReady ";
    std::unique_lock<std::mutex> lock(mutex_ready_);
    rpc_cond_.wait(lock, [=] { return !ready_; });
  }

 private:
  void SetStatus(bool ready) {
    VLOG(30) << "CollectiveServer SetReady ";
    {
      std::unique_lock<std::mutex> lock(mutex_ready_);
      ready_ = ready;
    }

    condition_ready_.notify_all();
  }

 private:
  std::shared_ptr<GatherGetHandler> get_handler_;
  std::shared_ptr<distributed::RPCServer> rpc_service_;
  std::shared_ptr<std::thread> server_thread_;
  // framework::Scope* scope_;

  std::mutex mutex_ready_;
  std::condition_variable condition_ready_;
  bool ready_{false};

  static std::once_flag init_flag_;
  static std::unique_ptr<CollectiveSever> collective_server_;

  friend RunServer(std::shared_ptr<distributed::RPCServer> service,
                   std::unique_ptr<CollectiveSever> server);
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
