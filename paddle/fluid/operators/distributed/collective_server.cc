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

#include <stdio.h>  // for removing the port file
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/operators/distributed/collective_server.h"

DECLARE_int32(rpc_get_thread_num);

namespace paddle {
namespace operators {
namespace distributed {
CollectiveServer::CollectiveServer(const std::string& end_point, int fan_in) {
  VLOG(1) << "Create colllective server:" << end_point << ", fan_in:" << fan_in;
  rpc_service_.reset(new RPCSERVER_T(end_point, fan_in));
}

void RunServer(std::shared_ptr<distributed::RPCServer> service,
               std::shared_ptr<CollectiveServer> server) {
  VLOG(1) << "Start colllective server";
  service->StartServer();
  service->WaitServerReady();

  service->SetCond(distributed::kRequestGet);

  while (true) {
    server->WaitInService();

    if (server->rpc_service_->IsExit()) {
      LOG(WARNING) << "get exit!rpc_processor break!";
      break;
    }

    service->WaitBarrier(distributed::kRequestGet);
    service->ResetBarrierCounter();
    server->SetSeriviceStatus(false);
  }
}

void CollectiveServer::StartServer() {
  get_handler_.reset(new distributed::GatherGetHandler());

  rpc_service_->RegisterRPC(distributed::kRequestGet, get_handler_.get(),
                            FLAGS_rpc_get_thread_num);

  server_thread_.reset(
      new std::thread(RunServer, rpc_service_, collective_server_));
  rpc_service_->WaitServerReady();
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
