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
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/listen_and_serv_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/nccl_helper.h"
#include "paddle/fluid/string/printf.h"

USE_NO_KERNEL_OP(listen_and_serv);

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;
namespace detail = paddle::operators::detail;
namespace string = paddle::string;

std::unique_ptr<detail::AsyncGRPCServer> rpc_service;

void StartServer(std::atomic<bool>* initialized) {
  f::Scope scope;
  p::CPUPlace place;
  scope.Var(NCCL_ID_VARNAME);
  p::DeviceContextPool& pool = p::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(p::CPUPlace());

  rpc_service.reset(new detail::AsyncGRPCServer("127.0.0.1:0", true));

  f::ProgramDesc empty_program;
  f::Executor executor(dev_ctx.GetPlace());
  rpc_service->SetScope(&scope);
  rpc_service->SetDevCtx(&dev_ctx);
  rpc_service->SetProgram(&empty_program);
  rpc_service->SetExecutor(&executor);

  std::thread server_thread(
      std::bind(&detail::AsyncGRPCServer::RunSyncUpdate, rpc_service.get()));
  *initialized = true;
  rpc_service->SetCond(0);
  auto recv = rpc_service->Get();
  LOG(INFO) << "got nccl id and stop server...";
  rpc_service->ShutDown();
  server_thread.join();
}

TEST(SendNcclId, DISABLED_Normal) {
  std::atomic<bool> initialized{false};
  std::thread server_thread(StartServer, &initialized);
  while (!initialized) {
  }
  // wait server to start
  // sleep(2);
  rpc_service->WaitServerReady();

  f::Scope scope;
  p::CPUPlace place;
  p::DeviceContextPool& pool = p::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(p::CPUPlace());

  auto var = scope.Var(NCCL_ID_VARNAME);
  // var->SetType(f::proto::VarType_Type_RAW);
  auto id = var->GetMutable<ncclUniqueId>();
  p::dynload::ncclGetUniqueId(id);

  int port = rpc_service->GetSelectedPort();
  std::string ep = string::Sprintf("127.0.0.1:%d", port);
  detail::RPCClient client;

  client.AsyncSendVariable(ep, dev_ctx, scope, NCCL_ID_VARNAME);
  client.Wait();
  server_thread.join();
  auto* ptr = rpc_service.release();
  delete ptr;
}
