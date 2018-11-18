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

#include <stdio.h>  // for removing the port file
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <thread>  // NOLINT
#include <vector>

#include "gflags/gflags.h"

#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/distributed/collective_server.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"

DECLARE_int32(rpc_get_thread_num, 5, "number of threads for rpc get");

namespace paddle {
namespace operators {
namespace distributed {
CollectiveSever::CollectiveSever(const std::string& end_point, int fan_in) {
  rpc_service_.reset(new RPCSERVER_T(end_point, fan_in));
}

static void RunServer(std::shared_ptr<distributed::RPCServer> service) {
  service->StartServer();
  VLOG(10) << "RunServer thread end";
}

void CollectiveSever::StartServer() {
  request_get_handler_.reset(
      new distributed::RequestGetHandler(sync_mode, dc_sgd));

  rpc_service_->RegisterRPC(distributed::kRequestGet,
                            request_get_handler_.get(),
                            FLAGS_rpc_get_thread_num);

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(RunServer, rpc_service_));
  VLOG(10) << "wait server thread to become ready...";
  rpc_service_->WaitServerReady();
}
};  // namespace distributed
};  // namespace operators
};  // namespace paddle
