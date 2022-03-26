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
#include <random>
#include <sstream>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/distributed/ps/service/heter_server.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::distributed;

void CreateVarsOnScope(framework::Scope* scope) {
  auto var1 = scope->Var("w");
  var1->GetMutable<phi::SelectedRows>();
  auto var2 = scope->Var("x");
  var2->GetMutable<framework::LoDTensor>();
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope);

  auto w = scope->Var("w")->GetMutable<phi::SelectedRows>();
  auto w_value = w->mutable_value();
  w_value->Resize({rows_numel, 10});
  for (int64_t i = 0; i < rows_numel; ++i) w->AutoGrownIndex(i, true);

  auto ptr = w_value->mutable_data<float>(*place);

  for (int64_t i = 0; i < w_value->numel(); ++i) {
    ptr[i] = static_cast<float>(i / 10);
  }

  auto x_var = scope->Var("x")->GetMutable<framework::LoDTensor>();
  float* x_ptr =
      x_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) {
    x_ptr[i] = 1.0;
  }
}

void StartSwitchServer(
    std::shared_ptr<distributed::HeterServer>& switch_server_ptr,  // NOLINT
    std::vector<std::string> endpoints,
    std::vector<std::string> peer_endpoints) {
  switch_server_ptr->SetPeerEndPoints(peer_endpoints);
  switch_server_ptr->SetEndPoint(endpoints[0]);
  switch_server_ptr->StartHeterService(false);
}

void StartSwitchInterServer(
    std::shared_ptr<distributed::HeterServer>& switch_server_ptr,  // NOLINT
    std::vector<std::string> endpoints,
    std::vector<std::string> peer_endpoints) {
  switch_server_ptr->SetPeerEndPoints(peer_endpoints);
  switch_server_ptr->SetInterEndpoint(endpoints[1]);
  switch_server_ptr->StartHeterInterService(false);
}

TEST(HETERSENDANDRECV, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);

  // 启动 switch server A & B
  std::string switch_a_endpoint("127.0.0.1:5000");
  std::string switch_a_endpoint_inter("127.0.0.1:5100");
  std::string switch_b_endpoint_inter("127.0.0.1:6100");
  std::string switch_b_endpoint("127.0.0.1:6000");

  std::shared_ptr<distributed::HeterServer> switch_server_ptr_a =
      std::make_shared<distributed::HeterServer>();
  std::vector<std::string> end_points{switch_a_endpoint};
  std::vector<std::string> peer_endpoints{switch_b_endpoint_inter};
  std::thread switch_server_a_thread(StartSwitchServer,
                                     std::ref(switch_server_ptr_a), end_points,
                                     peer_endpoints);
  switch_server_ptr_a->WaitServerReady();

  std::shared_ptr<distributed::HeterServer> switch_server_ptr_b =
      std::make_shared<distributed::HeterServer>();
  end_points = {switch_b_endpoint, switch_b_endpoint_inter};
  peer_endpoints = {};
  std::thread switch_server_b_thread(StartSwitchServer,
                                     std::ref(switch_server_ptr_b), end_points,
                                     peer_endpoints);
  switch_server_ptr_b->WaitServerReady();

  end_points = {switch_b_endpoint, switch_b_endpoint_inter};
  peer_endpoints = {};
  std::thread switch_server_b_thread_inter(StartSwitchInterServer,
                                           std::ref(switch_server_ptr_b),
                                           end_points, peer_endpoints);
  switch_server_ptr_b->WaitServerReady();

  // 获取 client 实例
  std::shared_ptr<distributed::HeterClient> heter_client_ptr_ =
      distributed::HeterClient::GetInstance(
          {switch_a_endpoint, switch_b_endpoint}, {}, 0);

  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  framework::Executor exe(place);

  framework::ProgramDesc program;
  exe.Prepare(program, 0);  // solve undefined symbol: tensor_table.cc
  std::shared_ptr<framework::Scope> send_scope_ptr =
      std::make_shared<framework::Scope>();
  int64_t rows_numel = 10;
  InitTensorsOnClient(send_scope_ptr.get(), &place, rows_numel);
  LOG(INFO) << "InitTensorsOnClient done";

  auto send_async = [&]() -> void {
    std::string message_name = "send";
    std::vector<std::string> send_var_names{"w", "x"};
    int ret = heter_client_ptr_->Send(ctx, *send_scope_ptr, message_name,
                                      send_var_names);
    if (!ret) {
      LOG(ERROR) << ">>>> worker send success";
    }
  };
  std::thread send_thread(send_async);

  std::string message_name = "recv";
  std::vector<std::string> recv_var_names{"w", "x"};
  std::shared_ptr<framework::Scope> recv_scope_ptr =
      std::make_shared<framework::Scope>();
  int ret = heter_client_ptr_->Recv(ctx, *recv_scope_ptr, message_name,
                                    recv_var_names);
  if (!ret && recv_scope_ptr->FindVar("w") && recv_scope_ptr->FindVar("x")) {
    LOG(INFO) << ">>>> worker recv success";
  } else {
    LOG(INFO) << "worker recv failed";
  }

  send_thread.join();
  /*
  heter_client_ptr_->Stop();
  LOG(INFO) << "heter client main thread joined";
  */
  switch_server_ptr_a->Stop();
  LOG(INFO) << "switch server A stopped";

  switch_server_ptr_b->Stop();
  LOG(INFO) << "switch server B stopped";

  switch_server_a_thread.join();
  LOG(INFO) << "switch_server_a_thread joined";

  switch_server_b_thread.join();
  LOG(INFO) << "switch_server_b_thread joined";

  switch_server_b_thread_inter.join();
  LOG(INFO) << "switch_server_b_thread_inter joined";
}
#endif
