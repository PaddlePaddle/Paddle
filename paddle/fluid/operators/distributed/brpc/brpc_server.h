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

#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <string>

#include "brpc/server.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/operators/distributed/send_recv.pb.h"

namespace paddle {
namespace operators {
namespace distributed {

class AsyncBRPCServer final : public RPCServer {
 public:
  explicit AsyncBRPCServer(const std::string& address, int client_num)
      : RPCServer(address, client_num), ready_(0) {}

  virtual ~AsyncBRPCServer() {}
  void StartServer() override;
  void WaitServerReady() override;

 private:
  void ShutDownImpl() override;

  brpc::Server server_;

  static constexpr int idle_timeout_s_ = -1;
  static constexpr int max_concurrency_ = 0;

  std::mutex mutex_ready_;
  std::condition_variable condition_ready_;
  int ready_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
