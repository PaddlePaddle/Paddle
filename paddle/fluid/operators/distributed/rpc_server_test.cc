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
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::operators::distributed;

USE_OP(lookup_table);

std::unique_ptr<distributed::RPCServer> g_rpc_service;
std::unique_ptr<distributed::RequestHandler> g_req_handler;

framework::BlockDesc* AppendPrefetchBlcok(framework::ProgramDesc* program) {
  auto root_block = program->MutableBlock(0);
  auto* block = program->AppendBlock(*root_block);

  framework::VariableNameMap input({{"W", {"w"}}, {"Ids", {"ids"}}});
  framework::VariableNameMap output({{"Output", {"out"}}});
  auto op = block->AppendOp();
  op->SetType("lookup_table");
  op->SetInput("W", {"w"});
  op->SetInput("Ids", {"ids"});
  op->SetOutput("Out", {"out"});

  auto& out = *root_block->Var("out");
  out.SetType(framework::proto::VarType::SELECTED_ROWS);
  out.SetShape({10, 10});

  return block;
}

void CreateVarsOnScope(framework::Scope* scope, platform::CPUPlace* place) {
  auto w_var = scope->Var("w");
  w_var->GetMutable<framework::SelectedRows>();

  auto out_var = scope->Var("out");
  out_var->GetMutable<framework::SelectedRows>();

  auto ids_var = scope->Var("ids");
  ids_var->GetMutable<framework::SelectedRows>();
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto ids_var = scope->Var("ids")->GetMutable<framework::SelectedRows>();
  auto rows = ids_var->mutable_rows();
  for (int64_t i = 0; i < rows_numel; ++i) rows->push_back(i * 2);
  ids_var->mutable_value()->Resize({rows_numel, 1});
  ids_var->mutable_value()->mutable_data<float>(*place);
}

void InitTensorsOnServer(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto w = scope->Var("w")->GetMutable<framework::SelectedRows>();
  auto rows = w->mutable_rows();
  for (int64_t i = 0; i < rows_numel; ++i) rows->push_back(i);
  auto w_value = w->mutable_value();
  w_value->Resize({rows_numel, 10});

  auto ptr = w_value->mutable_data<float>(*place);

  for (int64_t i = 0; i < w_value->numel(); ++i) {
    ptr[i] = static_cast<float>(i / 10);
  }
}

void StartServer() {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  framework::Executor exe(place);
  platform::CPUDeviceContext ctx(place);
  auto* block = AppendPrefetchBlcok(&program);
  std::string in_var_name("ids");
  std::vector<int> prefetch_block_ids{block->ID()};
  auto prepared = exe.Prepare(program, prefetch_block_ids);
  InitTensorsOnServer(&scope, &place, 10);

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      prefetch_var_name_to_prepared;
  prefetch_var_name_to_prepared[in_var_name] = prepared[0];
  g_req_handler->SetProgram(&program);
  g_req_handler->SetPrefetchPreparedCtx(&prefetch_var_name_to_prepared);
  g_req_handler->SetDevCtx(&ctx);
  g_req_handler->SetScope(&scope);
  g_req_handler->SetExecutor(&exe);

  g_rpc_service->RegisterRPC(distributed::kRequestPrefetch,
                             g_req_handler.get());
  g_req_handler->SetRPCServer(g_rpc_service.get());

  std::thread server_thread(
      std::bind(&distributed::RPCServer::StartServer, g_rpc_service.get()));

  server_thread.join();
}

TEST(PREFETCH, CPU) {
  g_req_handler.reset(new distributed::RequestPrefetchHandler(true));
  g_rpc_service.reset(new RPCSERVER_T("127.0.0.1:0", 1));
  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>();

  std::thread server_thread(StartServer);
  g_rpc_service->WaitServerReady();

  int port = g_rpc_service->GetSelectedPort();
  std::string ep = paddle::string::Sprintf("127.0.0.1:%d", port);

  framework::Scope scope;
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  {
    // create var on local scope
    int64_t rows_numel = 5;
    InitTensorsOnClient(&scope, &place, rows_numel);
    std::string in_var_name("ids");
    std::string out_var_name("out");

    client->AsyncPrefetchVar(ep, ctx, scope, in_var_name, out_var_name);
    client->Wait();
    auto var = scope.Var(out_var_name);
    auto value = var->GetMutable<framework::SelectedRows>()->value();
    auto ptr = value.mutable_data<float>(place);

    for (int64_t i = 0; i < rows_numel; ++i) {
      EXPECT_EQ(ptr[0 + i * value.dims()[1]], static_cast<float>(i * 2));
    }
  }

  g_rpc_service->ShutDown();
  server_thread.join();
  LOG(INFO) << "begin reset";
  g_rpc_service.reset(nullptr);
  g_req_handler.reset(nullptr);
}
