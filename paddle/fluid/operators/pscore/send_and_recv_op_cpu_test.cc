/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined PADDLE_WITH_PSCORE
#include <stdlib.h>
#include <memory>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/service/heter_client.h"
#include "paddle/fluid/distributed/service/heter_server.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::distributed;

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

USE_OP(scale);
USE_OP(send_and_recv);

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

void CreateVarsOnScope(framework::Scope* scope) {
  auto w_var = scope->Var("w");
  w_var->GetMutable<framework::SelectedRows>();

  auto out_var = scope->Var("out");
  out_var->GetMutable<framework::LoDTensor>();

  auto micro_var = scope->Var("microbatch_id");
  micro_var->GetMutable<framework::LoDTensor>();

  auto ids_var = scope->Var("ids");
  ids_var->GetMutable<framework::LoDTensor>();

  auto x_var = scope->Var("x");
  x_var->GetMutable<framework::LoDTensor>();

  auto res_var = scope->Var("res");
  res_var->GetMutable<framework::LoDTensor>();
}

void InitTensorsOnServer(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope);
  auto w = scope->Var("w")->GetMutable<framework::SelectedRows>();
  auto w_value = w->mutable_value();
  w_value->Resize({rows_numel, 10});
  for (int64_t i = 0; i < rows_numel; ++i) w->AutoGrownIndex(i, true);

  auto ptr = w_value->mutable_data<float>(*place);

  for (int64_t i = 0; i < w_value->numel(); ++i) {
    ptr[i] = static_cast<float>(i / 10);
  }
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope);
  auto ids_var = scope->Var("ids")->GetMutable<framework::LoDTensor>();
  int64_t* ids_ptr =
      ids_var->mutable_data<int64_t>(framework::DDim({rows_numel, 1}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) ids_ptr[i] = i * 2;

  auto micro_id_var =
      scope->Var("microbatch_id")->GetMutable<framework::LoDTensor>();
  float* micro_id_ptr =
      micro_id_var->mutable_data<float>(framework::DDim({1}), *place);
  micro_id_ptr[0] = 0;

  auto x_var = scope->Var("x")->GetMutable<framework::LoDTensor>();
  float* x_ptr =
      x_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) x_ptr[i] = 1.0;

  auto res_var = scope->Var("res")->GetMutable<framework::LoDTensor>();
  float* res_ptr =
      res_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) res_ptr[i] = 1.0;
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
  // std::string in_var_name2("y");
  std::vector<int> prefetch_block_ids{block->ID()};

  LOG(INFO) << "before InitTensorsOnServer";
  InitTensorsOnServer(&scope, &place, 10);
  LOG(INFO) << "end InitTensorsOnServer";

  std::shared_ptr<distributed::RequestSendAndRecvHandler> b_req_handler;
  b_req_handler.reset(new distributed::RequestSendAndRecvHandler());
  LOG(INFO) << "before SetDevCtx";
  b_req_handler->SetDevCtx(&ctx);
  LOG(INFO) << "before SetScope";
  b_req_handler->SetScope(&scope);
  LOG(INFO) << "before HeterServer::GetInstance";
  b_rpc_service = distributed::HeterServer::GetInstance();
  b_rpc_service->SetEndPoint(endpoint);
  LOG(INFO) << "before HeterServer::RegisterServiceHandler";
  b_rpc_service->RegisterServiceHandler(
      in_var_name, [&](const MultiVarMsg* request, MultiVarMsg* response,
                       brpc::Controller* cntl) -> int {
        return b_req_handler->Handle(request, response, cntl);
      });

  b_rpc_service->SetRequestHandler(b_req_handler);
  LOG(INFO) << "before HeterServer::RunServer";
  std::thread server_thread(std::bind(RunServer, b_rpc_service));

  server_thread.join();
}

TEST(SENDANDRECV, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  std::string endpoint = "127.0.0.1:4444";
  std::string previous_endpoint = "127.0.0.1:4444";
  LOG(INFO) << "before StartSendAndRecvServer";
  b_rpc_service = distributed::HeterServer::GetInstance();
  std::thread server_thread(StartSendAndRecvServer, endpoint);
  b_rpc_service->WaitServerReady();
  using MicroScope =
      std::unordered_map<int, std::shared_ptr<std::vector<framework::Scope*>>>;
  using MiniScope = std::unordered_map<int, framework::Scope*>;
  std::shared_ptr<MiniScope> mini_scopes(new MiniScope{});
  std::shared_ptr<MicroScope> micro_scopes(new MicroScope{});
  auto* mini_scope = new framework::Scope();
  (*mini_scopes)[0] = mini_scope;
  std::shared_ptr<std::vector<framework::Scope*>> micro_scope(
      new std::vector<framework::Scope*>{});
  auto* micro_scope_0 = &(mini_scope->NewScope());
  (*micro_scope).push_back(micro_scope_0);
  (*micro_scopes)[0] = micro_scope;
  b_rpc_service->SetMicroBatchScopes(micro_scopes);
  b_rpc_service->SetMiniBatchScopes(mini_scopes);

  using TaskQueue =
      std::unordered_map<int,
                         std::shared_ptr<::paddle::framework::BlockingQueue<
                             std::pair<std::string, int>>>>;
  using SharedTaskQueue = std::shared_ptr<std::unordered_map<
      int, std::shared_ptr<::paddle::framework::BlockingQueue<
               std::pair<std::string, int>>>>>;
  SharedTaskQueue task_queue_(new TaskQueue{});
  (*task_queue_)[0] = std::make_shared<
      ::paddle::framework::BlockingQueue<std::pair<std::string, int>>>();
  b_rpc_service->SetTaskQueue(task_queue_);

  LOG(INFO) << "before HeterClient::GetInstance";
  distributed::HeterClient* rpc_client =
      distributed::HeterClient::GetInstance({endpoint}, {previous_endpoint}, 0)
          .get();

  PADDLE_ENFORCE_NE(rpc_client, nullptr,
                    platform::errors::InvalidArgument(
                        "Client Start Fail, Check Your Code & Env"));

  framework::Scope* scope = (*micro_scope)[0];
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);

  framework::Executor exe(place);
  // create var on local scope
  int64_t rows_numel = 10;
  LOG(INFO) << "before InitTensorsOnClient";
  InitTensorsOnClient(scope, &place, rows_numel);
  std::string in_var_name("x");
  std::string micro_var_name("microbatch_id");
  // std::string out_var_name("res");
  std::vector<std::string> send_var{in_var_name, micro_var_name};
  std::vector<std::string> recv_var{};
  std::string mode_str("forward");

  LOG(INFO) << "add block & op1";
  framework::ProgramDesc program;
  auto root_block = program.MutableBlock(0);
  // op for forward
  framework::OpDesc* op = root_block->AppendOp();
  op->SetType("send_and_recv");
  LOG(INFO) << "op1 set input";
  op->SetInput("X", std::vector<std::string>({in_var_name}));
  op->SetOutput("Out", {});
  op->SetAttr("next_endpoints", std::vector<std::string>({endpoint}));
  op->SetAttr("previous_endpoints",
              std::vector<std::string>({previous_endpoint}));
  op->SetAttr("trainer_id", 0);
  op->SetAttr("mode", mode_str);
  op->SetAttr("message_name", in_var_name);
  op->SetAttr("send_var_name", send_var);
  op->SetAttr("recv_var_name", recv_var);

  std::string mode_str2("backward");
  // op for backward
  LOG(INFO) << "add op2";
  framework::OpDesc* op2 = root_block->AppendOp();
  op2->SetType("send_and_recv");
  LOG(INFO) << "op2 set input";
  op2->SetInput("X", std::vector<std::string>({in_var_name}));
  op2->SetOutput("Out", {});
  op2->SetAttr("next_endpoints", std::vector<std::string>({endpoint}));
  op2->SetAttr("previous_endpoints",
               std::vector<std::string>({previous_endpoint}));
  op2->SetAttr("trainer_id", 0);
  op2->SetAttr("mode", mode_str2);
  op2->SetAttr("message_name", in_var_name);
  op2->SetAttr("send_var_name", send_var);
  op2->SetAttr("recv_var_name", recv_var);

  LOG(INFO) << "exe before prepare";
  auto prepared = exe.Prepare(program, 0);
  LOG(INFO) << "exe after prepare";

  LOG(INFO) << "before RunPreparedContext";
  exe.RunPreparedContext(prepared.get(), scope, false);

  LOG(INFO) << "client wait for Pop";
  auto task = (*task_queue_)[0]->Pop();
  LOG(INFO) << "client get from task queue";
  PADDLE_ENFORCE_EQ(
      task.first, "x",
      platform::errors::InvalidArgument(
          "Recv message and Send message name not match, Check your Code"));

  auto task2 = (*task_queue_)[0]->Pop();
  PADDLE_ENFORCE_EQ(
      task2.first, "x",
      platform::errors::InvalidArgument(
          "Recv message and Send message name not match, Check your Code"));

  rpc_client->FinalizeWorker();
  b_rpc_service->Stop();
  LOG(INFO) << "end server Stop";
  server_thread.join();
  LOG(INFO) << "end server thread join";
}
#endif
