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

#include <random>
#include <sstream>

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/distributed/ps/service/heter_server.h"
#include "paddle/fluid/framework/op_registry.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::distributed;

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

USE_OP_ITSELF(scale);

std::string get_ip_port() {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist(4444, 25000);
  int port = dist(rng);
  std::string ip_port;
  std::stringstream temp_str;
  temp_str << "127.0.0.1:";
  temp_str << port;
  temp_str >> ip_port;
  return ip_port;
}

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
  w_var->GetMutable<phi::SelectedRows>();

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

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
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

void InitTensorsOnClient2(framework::Scope* scope, platform::CPUPlace* place,
                          int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto ids_var = scope->Var("ids")->GetMutable<framework::LoDTensor>();
  int64_t* ids_ptr =
      ids_var->mutable_data<int64_t>(framework::DDim({rows_numel, 1}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) ids_ptr[i] = i * 2;

  auto micro_id_var =
      scope->Var("microbatch_id")->GetMutable<framework::LoDTensor>();
  float* micro_id_ptr =
      micro_id_var->mutable_data<float>(framework::DDim({1}), *place);
  micro_id_ptr[0] = 1;

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
  auto w = scope->Var("w")->GetMutable<phi::SelectedRows>();
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
  std::string in_var_name2("y");
  std::vector<int> prefetch_block_ids{block->ID()};

  LOG(INFO) << "before InitTensorsOnServer";
  InitTensorsOnServer(&scope, &place, 10);
  LOG(INFO) << "end InitTensorsOnServer";

  std::shared_ptr<distributed::SendAndRecvVariableHandler> b_req_handler;
  b_req_handler.reset(new distributed::SendAndRecvVariableHandler());
  LOG(INFO) << "before SetDevCtx";
  b_req_handler->SetDevCtx(&ctx);
  LOG(INFO) << "before SetScope";
  b_req_handler->SetScope(&scope);
  LOG(INFO) << "before HeterServer::GetInstance";
  std::shared_ptr<distributed::HeterServer> heter_server_ptr_ =
      distributed::HeterServer::GetInstance();
  heter_server_ptr_->SetEndPoint(endpoint);
  LOG(INFO) << "before HeterServer::RegisterServiceHandler";
  heter_server_ptr_->RegisterServiceHandler(
      in_var_name, [&](const MultiVarMsg* request, MultiVarMsg* response,
                       brpc::Controller* cntl) -> int {
        return b_req_handler->Handle(request, response, cntl);
      });
  heter_server_ptr_->RegisterServiceHandler(
      in_var_name2, [&](const MultiVarMsg* request, MultiVarMsg* response,
                        brpc::Controller* cntl) -> int {
        return b_req_handler->Handle(request, response, cntl);
      });

  heter_server_ptr_->SetServiceHandler(b_req_handler);
  LOG(INFO) << "before HeterServer::RunServer";
  RunServer(heter_server_ptr_);
  // std::thread server_thread(std::bind(RunServer, heter_server_ptr_));

  // server_thread.join();
}

TEST(SENDANDRECV, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  std::string endpoint = get_ip_port();
  std::string previous_endpoint = endpoint;
  LOG(INFO) << "before StartSendAndRecvServer";
  std::shared_ptr<distributed::HeterServer> heter_server_ptr_ =
      distributed::HeterServer::GetInstance();
  std::thread server_thread(StartSendAndRecvServer, endpoint);
  heter_server_ptr_->WaitServerReady();
  using MicroScope =
      std::unordered_map<int, std::shared_ptr<std::vector<framework::Scope*>>>;
  using MiniScope = std::unordered_map<int, framework::Scope*>;
  std::shared_ptr<MiniScope> mini_scopes(new MiniScope{});
  std::shared_ptr<MicroScope> micro_scopes(new MicroScope{});
  std::shared_ptr<std::vector<framework::Scope*>> micro_scope(
      new std::vector<framework::Scope*>{});
  auto* mini_scope = new framework::Scope();
  (*mini_scopes)[0] = mini_scope;
  auto* micro_scope_0 = &(mini_scope->NewScope());
  auto* micro_scope_1 = &(mini_scope->NewScope());
  (*micro_scope).push_back(micro_scope_0);
  (*micro_scope).push_back(micro_scope_1);
  (*micro_scopes)[0] = micro_scope;
  heter_server_ptr_->SetMicroBatchScopes(micro_scopes);
  heter_server_ptr_->SetMiniBatchScopes(mini_scopes);

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
  heter_server_ptr_->SetTaskQueue(task_queue_);

  LOG(INFO) << "before HeterClient::GetInstance";
  distributed::HeterClient* heter_client_ptr_ =
      distributed::HeterClient::GetInstance({endpoint}, {previous_endpoint}, 0)
          .get();

  framework::Scope* scope = (*micro_scope)[0];
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);

  // create var on local scope
  int64_t rows_numel = 10;
  LOG(INFO) << "before InitTensorsOnClient";
  InitTensorsOnClient(scope, &place, rows_numel);
  std::string in_var_name("x");
  std::string micro_var_name("microbatch_id");
  std::string out_var_name("res");
  std::vector<std::string> send_var = {in_var_name, micro_var_name};
  std::vector<std::string> recv_var = {};

  LOG(INFO) << "before SendAndRecvAsync";
  heter_client_ptr_->SendAndRecvAsync(ctx, *scope, in_var_name, send_var,
                                      recv_var, "forward");

  LOG(INFO) << "client wait for Pop";
  auto task = (*task_queue_)[0]->Pop();
  LOG(INFO) << "client get from task queue";
  PADDLE_ENFORCE_EQ(
      task.first, "x",
      platform::errors::InvalidArgument(
          "Recv message and Send message name not match, Check your Code"));

  InitTensorsOnClient2((*micro_scope)[1], &place, rows_numel);
  LOG(INFO) << "before SendAndRecvAsync 2";
  std::string in_var_name2("y");
  heter_client_ptr_->SendAndRecvAsync(ctx, *((*micro_scope)[1]), in_var_name2,
                                      send_var, recv_var, "backward");
  LOG(INFO) << "after SendAndRecvAsync 2";

  auto task2 = (*task_queue_)[0]->Pop();
  PADDLE_ENFORCE_EQ(
      task2.first, "y",
      platform::errors::InvalidArgument(
          "Recv message and Send message name not match, Check your Code"));

  heter_server_ptr_->Stop();
  LOG(INFO) << "end server Stop";
  server_thread.join();
  LOG(INFO) << "end server thread join";
}
