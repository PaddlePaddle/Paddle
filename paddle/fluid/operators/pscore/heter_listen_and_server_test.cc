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
#include "paddle/fluid/distributed/service/heter_client.h"
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

USE_OP(scale);
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
  LOG(INFO) << "before StartSendAndRecvServer";
  FLAGS_eager_delete_tensor_gb = -1;
  std::thread server_thread(StartHeterServer);
  sleep(1);

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
  rpc_client->Stop();
  LOG(INFO) << "end server Stop";
  server_thread.join();
  LOG(INFO) << "end server thread join";
}
