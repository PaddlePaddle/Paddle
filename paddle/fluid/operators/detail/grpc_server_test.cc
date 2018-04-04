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
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/detail/grpc_server.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace detail = paddle::operators::detail;

USE_OP(lookup_table);

std::unique_ptr<detail::AsyncGRPCServer> rpc_service_;

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
  out.SetType(framework::proto::VarType::LOD_TENSOR);
  out.SetShape({10, 10});

  return block;
}

void CreateVarsOnScope(framework::Scope* scope, platform::CPUPlace* place) {
  auto w_var = scope->Var("w");
  auto w = w_var->GetMutable<framework::LoDTensor>();
  w->Resize({10, 10});
  w->mutable_data<float>(*place);

  auto out_var = scope->Var("out");
  auto out = out_var->GetMutable<framework::LoDTensor>();
  out->Resize({5, 10});
  out->mutable_data<float>(*place);

  auto ids_var = scope->Var("ids");
  auto ids = ids_var->GetMutable<framework::LoDTensor>();
  ids->Resize({5, 1});
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place) {
  CreateVarsOnScope(scope, place);
  auto ids = scope->Var("ids")->GetMutable<framework::LoDTensor>();
  auto ptr = ids->mutable_data<int64_t>(*place);
  for (int64_t i = 0; i < ids->numel(); ++i) {
    ptr[i] = i * 2;
  }
}

void InitTensorsOnServer(framework::Scope* scope, platform::CPUPlace* place) {
  CreateVarsOnScope(scope, place);
  auto w_var = scope->Var("w");
  auto w = w_var->GetMutable<framework::LoDTensor>();
  auto ptr = w->mutable_data<float>(*place);
  for (int64_t i = 0; i < w->numel(); ++i) {
    ptr[i] = static_cast<float>(i / 10);
  }
}

void StartServer(const std::string& endpoint) {
  rpc_service_.reset(new detail::AsyncGRPCServer(endpoint));
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  framework::Executor exe(place);
  platform::CPUDeviceContext ctx(place);
  auto* block = AppendPrefetchBlcok(&program);
  InitTensorsOnServer(&scope, &place);

  rpc_service_->SetProgram(&program);
  rpc_service_->SetPrefetchBlkdId(block->ID());
  rpc_service_->SetDevCtx(&ctx);
  rpc_service_->SetScope(&scope);
  rpc_service_->SetExecutor(&exe);

  rpc_service_->RunSyncUpdate();
}

TEST(PREFETCH, CPU) {
  // start up a server instance backend
  // TODO(Yancey1989): Need to start a server with optimize blocks and
  // prefetch blocks.
  std::thread server_thread(StartServer, "127.0.0.1:8889");
  sleep(2);
  framework::Scope scope;
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  // create var on local scope
  InitTensorsOnClient(&scope, &place);
  std::string in_var_name("ids");
  std::string out_var_name("out");

  detail::RPCClient client;
  client.AsyncPrefetchVariable("127.0.0.1:8889", ctx, scope, in_var_name,
                               out_var_name);
  client.Wait();

  auto out_var = scope.Var(out_var_name);
  auto out = out_var->Get<framework::LoDTensor>();

  auto out_ptr = out.data<float>();
  rpc_service_->ShutDown();
  server_thread.join();
  rpc_service_.reset(nullptr);

  EXPECT_EQ(out.dims().size(), 2);
  EXPECT_EQ(out_ptr[0], static_cast<float>(0));
  EXPECT_EQ(out_ptr[0 + 1 * out.dims()[1]], static_cast<float>(2));
  EXPECT_EQ(out_ptr[0 + 2 * out.dims()[1]], static_cast<float>(4));
  EXPECT_EQ(out_ptr[0 + 3 * out.dims()[1]], static_cast<float>(6));
  EXPECT_EQ(out_ptr[0 + 4 * out.dims()[1]], static_cast<float>(8));
}
