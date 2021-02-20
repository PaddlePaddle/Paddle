/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdlib.h>
#include <memory>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/service/heter_client.h"
#include "paddle/fluid/distributed/service/heter_server.h"
namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::distributed;

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

USE_OP(scale);

std::shared_ptr<distributed::HeterServer> b_rpc_service;

framework::BlockDesc* AppendSendAndRecvBlock(framework::ProgramDesc* program) {
  auto root_block = program->MutableBlock(0);
  auto* block = program->AppendBlock(*root_block);

  framework::OpDesc* op = block->AppendOp();
  op->SetType("scale");
  op->SetInput("X", {"x"});
  op->SetOutput("Out", {"res"});
  op->SetAttr("scale", 0.5f);

  auto& out = *root_block->Var("res");
  out.SetType(framework::proto::VarType::LOD_TENSOR);
  out.SetShape({1, 10});

  return block;
}

void CreateVarsOnScope(framework::Scope* scope, platform::CPUPlace* place) {
  auto w_var = scope->Var("w");
  w_var->GetMutable<framework::SelectedRows>();

  auto out_var = scope->Var("out");
  out_var->GetMutable<framework::LoDTensor>();

  auto ids_var = scope->Var("ids");
  ids_var->GetMutable<framework::LoDTensor>();

  auto x_var = scope->Var("x");
  x_var->GetMutable<framework::LoDTensor>();

  auto res_var = scope->Var("res");
  res_var->GetMutable<framework::LoDTensor>();
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto ids_var = scope->Var("ids")->GetMutable<framework::LoDTensor>();
  int64_t* ids_ptr =
      ids_var->mutable_data<int64_t>(framework::DDim({rows_numel, 1}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) ids_ptr[i] = i * 2;

  auto x_var = scope->Var("x")->GetMutable<framework::LoDTensor>();
  float* x_ptr =
      x_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) x_ptr[i] = 1.0;

  auto res_var = scope->Var("res")->GetMutable<framework::LoDTensor>();
  float* res_ptr =
      res_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) res_ptr[i] = 1.0;
}

void InitTensorsOnServer(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto w = scope->Var("w")->GetMutable<framework::SelectedRows>();
  auto w_value = w->mutable_value();
  w_value->Resize({rows_numel, 10});
  for (int64_t i = 0; i < rows_numel; ++i) w->AutoGrownIndex(i, true);

  auto ptr = w_value->mutable_data<float>(*place);

  for (int64_t i = 0; i < w_value->numel(); ++i) {
    ptr[i] = static_cast<float>(i / 10);
  }
}

void RunServer(std::shared_ptr<paddle::distributed::HeterServer> service) {
  service->StartHeterService();
}

void StartSendAndRecvServer(std::string endpoint) {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  framework::Executor exe(place);
  platform::CPUDeviceContext ctx(place);
  LOG(INFO) << "before AppendSendAndRecvBlock";
  auto block = AppendSendAndRecvBlock(&program);
  std::string in_var_name("x");
  std::vector<int> prefetch_block_ids{block->ID()};
  auto prepared = exe.Prepare(program, prefetch_block_ids);

  LOG(INFO) << "before InitTensorsOnServer";
  InitTensorsOnServer(&scope, &place, 10);
  LOG(INFO) << "end InitTensorsOnServer";
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      message_to_prepared_ctx;
  message_to_prepared_ctx[in_var_name] = prepared[0];

  std::shared_ptr<distributed::RequestSendAndRecvHandler> b_req_handler;
  b_req_handler.reset(new distributed::RequestSendAndRecvHandler());
  LOG(INFO) << "before SetProgram";
  b_req_handler->SetProgram(&program);
  LOG(INFO) << "before SetGradToPreparedCtx";
  b_req_handler->SetGradToPreparedCtx(&message_to_prepared_ctx);
  LOG(INFO) << "before SetDevCtx";
  b_req_handler->SetDevCtx(&ctx);
  LOG(INFO) << "before SetScope";
  b_req_handler->SetScope(&scope);
  LOG(INFO) << "before SetExecutor";
  b_req_handler->SetExecutor(&exe);
  LOG(INFO) << "before HeterServer::GetInstance";
  b_rpc_service = distributed::HeterServer::GetInstance();
  b_rpc_service->SetEndPoint(endpoint);
  LOG(INFO) << "before HeterServer::RegisterServiceHandler";
  b_rpc_service->RegisterServiceHandler(
      in_var_name, [&](const MultiVarMsg* request, MultiVarMsg* response,
                       brpc::Controller* cntl) -> int {
        return b_req_handler->Handle(request, response, cntl);
      });

  LOG(INFO) << "before HeterServer::RunServer";
  std::thread server_thread(std::bind(RunServer, b_rpc_service));

  server_thread.join();
}

TEST(SENDANDRECV, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  std::string endpoint = "127.0.0.1:4444";
  LOG(INFO) << "before StartSendAndRecvServer";
  b_rpc_service = distributed::HeterServer::GetInstance();
  std::thread server_thread(StartSendAndRecvServer, endpoint);
  b_rpc_service->WaitServerReady();

  LOG(INFO) << "before HeterClient::GetInstance";
  distributed::HeterClient* rpc_client =
      distributed::HeterClient::GetInstance({endpoint}, 0).get();

  PADDLE_ENFORCE_NE(rpc_client, nullptr,
                    platform::errors::InvalidArgument(
                        "Client Start Fail, Check Your Code & Env"));

  framework::Scope scope;
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);

  // create var on local scope
  int64_t rows_numel = 10;
  LOG(INFO) << "before InitTensorsOnClient";
  InitTensorsOnClient(&scope, &place, rows_numel);
  std::string in_var_name("x");
  std::string out_var_name("res");
  std::vector<std::string> send_var = {in_var_name};
  std::vector<std::string> recv_var = {out_var_name};

  LOG(INFO) << "before SendAndRecvAsync";
  rpc_client->SendAndRecvAsync({endpoint}, ctx, scope, in_var_name, send_var,
                               recv_var);
  auto var = scope.Var(out_var_name);
  auto value = var->GetMutable<framework::LoDTensor>();
  auto ptr = value->mutable_data<float>(place);

  LOG(INFO) << "before CHECK";
  for (int64_t i = 0; i < rows_numel; ++i) {
    LOG(INFO) << "ptr " << i << " is " << ptr[i];
    EXPECT_EQ(ptr[i], 0.5);
  }
  LOG(INFO) << "end CHECK";
  rpc_client->FinalizeWorker();
  // b_rpc_service->Stop();
  b_rpc_service->Stop();
  LOG(INFO) << "end server Stop";
  server_thread.join();
  LOG(INFO) << "end server thread join";
}
