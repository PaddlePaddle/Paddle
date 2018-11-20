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
#include "paddle/fluid/operators/distributed/collective_client.h"
#include "paddle/fluid/operators/distributed/collective_server.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::operators::distributed;

std::shared_ptr<distributed::CollectiveServer> StartServer(
    const std::string& ep, int fan_in, framework::Scope* scope,
    platform::DeviceContext* dev_ctx) {
  distributed::CollectiveServer* server =
      distributed::CollectiveServer::GetInstance(ep, fan_in);

  auto rpc_server = server->GetRPCServer();
  rpc_server->RegisterVar("var1", distributed::kRequestGetMonomerVariable,
                          scope, dev_ctx);

  std::cout << "StartServer return" << std::endl;
  return std::shared_ptr<distributed::CollectiveServer>(server);
}

std::shared_ptr<framework::Scope> GenerateVars(platform::Place place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);

  framework::Scope* scope = new framework::Scope();
  framework::Variable* var = scope->Var("var1");
  auto* slr = var->GetMutable<framework::SelectedRows>();
  slr->set_height(1000);

  auto* tensor = slr->mutable_value();
  auto* rows = slr->mutable_rows();

  tensor->Resize(framework::make_ddim({3, 5}));
  tensor->mutable_data<float>(place);

  paddle::operators::math::set_constant(ctx, tensor, 32.7);
  for (int i = 0; i < 3; ++i) rows->push_back(i);

  std::cout << "src:" << slr->Info();

  return std::shared_ptr<framework::Scope>(scope);
}

void Gather(std::vector<std::string> eps, platform::DeviceContext* dev_ctx) {
  distributed::CollectiveClient* client =
      distributed::CollectiveClient::GetInstance();

  framework::Scope* scope = new framework::Scope();
  framework::Variable* var = scope->Var("var1");
  var->GetMutable<framework::SelectedRows>();

  std::vector<const framework::SelectedRows*> dst;
  client->Gather(eps, *dev_ctx, *scope, "var1", &dst);
  std::cout << "dst:" << dst[0]->Info();
}

TEST(PREFETCH, GPU) {
  platform::CUDAPlace place;
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);

  std::string ep = "127.0.0.1:7164";
  auto scope = GenerateVars(place);

  auto* v1 = scope->FindVar("var1");
  std::cout << "var1:" << v1 << std::endl;

  auto server = StartServer(ep, 2, scope.get(), &ctx);
  auto rpc_server = server->GetRPCServer();

  std::vector<std::string> eps{ep};
  Gather(eps, &ctx);
  Gather(eps, &ctx);

  std::cout << "begin WaitVarBarrier" << std::endl;
  rpc_server->WaitVarBarrier("var1");
  server->Stop();
}
