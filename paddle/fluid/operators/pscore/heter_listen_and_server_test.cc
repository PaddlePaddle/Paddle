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
#include <unistd.h>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/distributed/ps/service/heter_server.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::distributed;

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;
DECLARE_double(eager_delete_tensor_gb);

USE_OP_ITSELF(scale);
USE_NO_KERNEL_OP(heter_listen_and_serv);

framework::BlockDesc* AppendSendAndRecvBlock(framework::ProgramDesc* program) {
  framework::BlockDesc* block =
      program->AppendBlock(*(program->MutableBlock(0)));

  framework::OpDesc* op = block->AppendOp();
  op->SetType("scale");
  op->SetInput("X", {"x"});
  op->SetOutput("Out", {"res"});
  op->SetAttr("scale", 0.5f);

  auto* out = block->Var("res");
  out->SetType(framework::proto::VarType::LOD_TENSOR);
  out->SetShape({1, 10});

  return block;
}

void GetHeterListenAndServProgram(framework::ProgramDesc* program) {
  auto root_block = program->MutableBlock(0);

  auto* sub_block = AppendSendAndRecvBlock(program);
  std::vector<framework::BlockDesc*> optimize_blocks;
  optimize_blocks.push_back(sub_block);

  std::vector<std::string> message_to_block_id = {"x:1"};
  std::string endpoint = "127.0.0.1:19944";

  framework::OpDesc* op = root_block->AppendOp();
  op->SetType("heter_listen_and_serv");
  op->SetInput("X", {});
  op->SetAttr("message_to_block_id", message_to_block_id);
  op->SetAttr("optimize_blocks", optimize_blocks);
  op->SetAttr("endpoint", endpoint);
  op->SetAttr("fanin", 1);
  op->SetAttr("pserver_id", 0);
}

void CreateVarsOnScope(framework::Scope* scope, platform::CPUPlace* place) {
  auto x_var = scope->Var("x");
  x_var->GetMutable<framework::LoDTensor>();

  auto micro_var = scope->Var("microbatch_id");
  micro_var->GetMutable<framework::LoDTensor>();

  auto res_var = scope->Var("res");
  res_var->GetMutable<framework::LoDTensor>();
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto x_var = scope->Var("x")->GetMutable<framework::LoDTensor>();
  float* x_ptr =
      x_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) x_ptr[i] = 1.0;

  auto micro_id_var =
      scope->Var("microbatch_id")->GetMutable<framework::LoDTensor>();
  float* micro_id_ptr =
      micro_id_var->mutable_data<float>(framework::DDim({1}), *place);
  micro_id_ptr[0] = 0;

  auto res_var = scope->Var("res")->GetMutable<framework::LoDTensor>();
  float* res_ptr =
      res_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) res_ptr[i] = 1.0;
}

void InitTensorsOnClient2(framework::Scope* scope, platform::CPUPlace* place,
                          int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto x_var = scope->Var("x")->GetMutable<framework::LoDTensor>();
  float* x_ptr =
      x_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) x_ptr[i] = 1.0;

  auto micro_id_var =
      scope->Var("microbatch_id")->GetMutable<framework::LoDTensor>();
  float* micro_id_ptr =
      micro_id_var->mutable_data<float>(framework::DDim({1}), *place);
  micro_id_ptr[0] = 1;

  auto res_var = scope->Var("res")->GetMutable<framework::LoDTensor>();
  float* res_ptr =
      res_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) res_ptr[i] = 1.0;
}

void InitTensorsOnServer(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
}

void StartHeterServer() {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  framework::Executor exe(place);
  platform::CPUDeviceContext ctx(place);

  LOG(INFO) << "before GetHeterListenAndServProgram";
  GetHeterListenAndServProgram(&program);
  auto prepared = exe.Prepare(program, 0);

  LOG(INFO) << "before InitTensorsOnServer";
  InitTensorsOnServer(&scope, &place, 10);

  LOG(INFO) << "before RunPreparedContext";
  exe.RunPreparedContext(prepared.get(), &scope, false);
}

TEST(HETER_LISTEN_AND_SERV, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  std::string endpoint = "127.0.0.1:19944";
  std::string previous_endpoint = "127.0.0.1:19944";
  LOG(INFO) << "before StartSendAndRecvServer";
  FLAGS_eager_delete_tensor_gb = -1;
  std::thread server_thread(StartHeterServer);
  sleep(1);

  auto b_rpc_service = distributed::HeterServer::GetInstance();
  b_rpc_service->WaitServerReady();
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
  rpc_client->SendAndRecvAsync(ctx, *scope, in_var_name, send_var, recv_var,
                               "forward");
  auto task = (*task_queue_)[0]->Pop();
  PADDLE_ENFORCE_EQ(
      task.first, "x",
      platform::errors::InvalidArgument(
          "Recv message and Send message name not match, Check your Code"));

  InitTensorsOnClient2((*micro_scope)[1], &place, rows_numel);
  LOG(INFO) << "before SendAndRecvAsync 2";
  rpc_client->SendAndRecvAsync(ctx, *((*micro_scope)[1]), in_var_name, send_var,
                               recv_var, "backward");
  auto task2 = (*task_queue_)[0]->Pop();
  PADDLE_ENFORCE_EQ(
      task2.first, "x",
      platform::errors::InvalidArgument(
          "Recv message and Send message name not match, Check your Code"));

  rpc_client->Stop();
  LOG(INFO) << "end server Stop";
  server_thread.join();
  LOG(INFO) << "end server thread join";
}
