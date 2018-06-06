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

#include "brpc/server.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/request_handler.h"
#include "paddle/fluid/operators/detail/rpc_server.h"
#include "paddle/fluid/operators/detail/send_recv.pb.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace detail {

class AsyncBRPCServer final : public RPCServer {
 public:
  explicit AsyncBRPCServer(const std::string& address, int client_num)
      : RPCServer(address, client_num) {}

  virtual ~AsyncBRPCServer() {}
  void StartServer() override;
  void WaitServerReady() override;

 private:
  void ShutDownImpl() override;

  brpc::Server server_;

  static constexpr int idle_timeout_s_ = -1;
  static constexpr int max_concurrency_ = 0;
};

};  // namespace detail
};  // namespace operators
};  // namespace paddle
