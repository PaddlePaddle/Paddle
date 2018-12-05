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

std::unique_ptr<distributed::CollectiveServer> StartServer(
    const std::string& ep, int fan_in, framework::Scope* scope,
    const platform::Place& place) {
  distributed::CollectiveServer* server =
      distributed::CollectiveServer::GetInstance(ep, fan_in);
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);

  auto rpc_server = server->GetRPCServer();
  rpc_server->RegisterVar("var1", distributed::kRequestGetMonomerVariable,
                          scope, dev_ctx);

  std::cout << "StartServer return" << std::endl;
  return std::unique_ptr<distributed::CollectiveServer>(server);
}

void GenerateVars(framework::Scope* scope, platform::Place place,
                  const std::string& var_name) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);

  framework::Variable* var = scope->Var("var1");
  auto* slr = var->GetMutable<framework::SelectedRows>();
  slr->set_height(1000);

  auto* tensor = slr->mutable_value();
  auto* rows = slr->mutable_rows();

  tensor->Resize(framework::make_ddim({3, 5}));
  tensor->mutable_data<float>(place);

  paddle::operators::math::set_constant(ctx, tensor, 32.7);
  for (int i = 0; i < 3; ++i) rows->push_back(i);

  std::cout << "generated src:" << distributed::GetSelectedRowsInfo(*slr)
            << std::endl;
}

TEST(PREFETCH, GPU) {
  platform::CUDAPlace place;

  std::string ep = "127.0.0.1:7164";
  std::string var_name = "var1";

  // prepare server
  framework::Scope* server_scope = new framework::Scope();
  GenerateVars(server_scope, place, var_name);
  auto server = StartServer(ep, 1, server_scope, place);
  auto rpc_server = server->GetRPCServer();

  // prepare client
  framework::Scope* client_scope = new framework::Scope();
  GenerateVars(client_scope, place, var_name);

  std::vector<std::string> eps{ep};
  std::string dst_var_name = var_name + "_dst_";
  distributed::CollectiveClient::ReduceSelectedRows<float>(
      eps, var_name, client_scope, dst_var_name);
  auto slr = client_scope->FindVar(dst_var_name)
                 ->GetMutable<framework::SelectedRows>();
  std::cout << "ReduceSelectedRows:" << distributed::GetSelectedRowsInfo(*slr)
            << std::endl;

  std::cout << "begin WaitVarBarrier" << std::endl;
  rpc_server->WaitVarBarrier(dst_var_name);
  rpc_server->ClearRegisteredVars();
  server->Stop();

  server.release();
  delete server_scope;
  delete client_scope;
}
