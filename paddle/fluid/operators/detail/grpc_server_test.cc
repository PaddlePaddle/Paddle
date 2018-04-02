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
#include <thread>

#include "gtest/gtest.h"
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/detail/grpc_server.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace detail = paddle::operators::detail;

std::unique_ptr<detail::AsyncGRPCServer> rpc_service_;

void StartServer(const std::string& endpoint) {
  rpc_service_.reset(new detail::AsyncGRPCServer(endpoint));
  rpc_service_->RunSyncUpdate();
}

TEST(PREFETCH, CPU) {
  // start up a server instance backend
  // TODO(Yancey1989): Need to start a server with optimize blocks and
  // prefetch blocks.
  std::thread server_thread(StartServer, "127.0.0.1:8889");
  framework::Scope scope;
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  // create var on local scope
  std::string in_var_name("in");
  std::string out_var_name("out");
  auto* in_var = scope.Var(in_var_name);
  auto* in_tensor = in_var->GetMutable<framework::LoDTensor>();
  in_tensor->Resize({10, 10});
  VLOG(3) << "before mutable_data";
  in_tensor->mutable_data<int>(place);

  scope.Var(out_var_name);

  VLOG(3) << "before fetch";
  detail::RPCClient client;
  client.AsyncPrefetchVariable("127.0.0.1:8889", ctx, scope, in_var_name,
                               out_var_name);
  client.Wait();

  rpc_service_->ShutDown();
  server_thread.join();
  rpc_service_.reset(nullptr);
}
