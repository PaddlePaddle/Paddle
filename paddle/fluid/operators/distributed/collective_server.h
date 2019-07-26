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
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "gflags/gflags.h"

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/handlers/get_monomer_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace operators {
namespace distributed {

class CollectiveServer final {
 public:
  // NOTE: CollectiveServer will serv on the scope and dev_ctx passed here.
  explicit CollectiveServer(const std::string &end_point, int fan_in,
                            framework::Scope *scope,
                            platform::DeviceContext *dev_ctx);

  virtual ~CollectiveServer() {}

  void StartServer();

  static CollectiveServer *GetInstance(const std::string &end_point, int fan_in,
                                       framework::Scope *scope,
                                       platform::DeviceContext *dev_ctx) {
    std::call_once(init_flag_, [&]() {
      if (collective_server_.get() == nullptr) {
        collective_server_.reset(
            new CollectiveServer(end_point, fan_in, scope, dev_ctx));
        collective_server_->StartServer();
      }
    });

    return collective_server_.get();
  }

  std::shared_ptr<RPCServer> GetRPCServer() { return rpc_server_; }

  void Stop();

 private:
  std::unique_ptr<GetMonomerHandler> get_monomer_handler_;
  std::unique_ptr<GetMonomerBarrierHandler> get_barrier_handler_;

  std::shared_ptr<distributed::RPCServer> rpc_server_;
  std::shared_ptr<std::thread> server_thread_;
  std::shared_ptr<std::thread> loop_thread_;

  bool ready_{false};

  static std::once_flag init_flag_;
  static std::shared_ptr<CollectiveServer> collective_server_;

  // ***NOT OWNED***
  framework::Scope *scope_;
  platform::DeviceContext *dev_ctx_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
