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

#include <unistd.h>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/listen_and_serv_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/nccl_helper.h"
#include "paddle/fluid/string/printf.h"

#ifdef PADDLE_WITH_GRPC
#include "paddle/fluid/operators/send_recv_util.h"
#endif

USE_NO_KERNEL_OP(listen_and_serv);

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;
namespace distributed = paddle::operators::distributed;
namespace string = paddle::string;

std::unique_ptr<distributed::RPCServer> g_rpc_service;
std::unique_ptr<distributed::RequestHandler> g_req_handler;

void StartServer() {
  f::Scope scope;
  p::CPUPlace place;
  scope.Var(NCCL_ID_VARNAME);
  p::DeviceContextPool& pool = p::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(p::CPUPlace());

  f::ProgramDesc empty_program;
  f::Executor executor(dev_ctx.GetPlace());
  g_req_handler->SetScope(&scope);
  g_req_handler->SetDevCtx(&dev_ctx);
  g_req_handler->SetProgram(&empty_program);
  g_req_handler->SetExecutor(&executor);

  g_rpc_service->RegisterRPC(distributed::kRequestSend, g_req_handler.get());
  g_req_handler->SetRPCServer(g_rpc_service.get());

  std::thread server_thread(
      std::bind(&distributed::RPCServer::StartServer, g_rpc_service.get()));

  g_rpc_service->SetCond(distributed::kRequestSend);
  g_rpc_service->WaitBarrier(distributed::kRequestSend);

  LOG(INFO) << "got nccl id and stop server...";
  g_rpc_service->ShutDown();
  server_thread.join();
}

TEST(SendNcclId, RPCServer) {
  g_req_handler.reset(new distributed::RequestSendHandler(true));
  g_rpc_service.reset(new RPCSERVER_T("127.0.0.1:0", 1));

  std::thread server_thread(StartServer);
  g_rpc_service->WaitServerReady();

  f::Scope scope;
  p::CPUPlace place;
  p::DeviceContextPool& pool = p::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(p::CPUPlace());

  auto var = scope.Var(NCCL_ID_VARNAME);
  auto id = var->GetMutable<ncclUniqueId>();
  p::dynload::ncclGetUniqueId(id);

  int port = g_rpc_service->GetSelectedPort();

  std::string ep = string::Sprintf("127.0.0.1:%d", port);

  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>();

  LOG(INFO) << "connect to server" << ep;
  client->AsyncSendVar(ep, dev_ctx, scope, NCCL_ID_VARNAME);
  client->Wait();
  client->AsyncSendBatchBarrier(ep);
  client->Wait();

  server_thread.join();
  g_rpc_service.reset(nullptr);
  g_req_handler.reset(nullptr);
}
