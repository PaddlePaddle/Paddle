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

#include <cmath>
#include <fstream>
#include <iostream>
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

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;

void CreateVarsOnScope(framework::Scope* scope) {
  auto var1 = scope->Var("w");
  var1->GetMutable<phi::SelectedRows>();
  auto var2 = scope->Var("x");
  var2->GetMutable<phi::DenseTensor>();
}

void InitTensorsOnClient(framework::Scope* scope,
                         phi::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope);

  auto w = scope->Var("w")->GetMutable<phi::SelectedRows>();
  auto w_value = w->mutable_value();
  w_value->Resize({rows_numel, 10});
  for (int64_t i = 0; i < rows_numel; ++i) w->AutoGrownIndex(i, true);

  auto ptr = w_value->mutable_data<float>(*place);

  for (int64_t i = 0; i < w_value->numel(); ++i) {
    ptr[i] = static_cast<float>(i) / 10.0;
  }

  auto x_var = scope->Var("x")->GetMutable<phi::DenseTensor>();
  float* x_ptr = x_var->mutable_data<float>(phi::DDim({1, rows_numel}), *place);
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

void TestShardSendRecv(
    std::shared_ptr<distributed::HeterClient> heter_client_ptr_) {
  auto send_async = [&]() -> void {
    std::vector<int64_t> vars_len{2 * sizeof(float),
                                  4 * sizeof(float)};  // 字节数
    std::vector<float> values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int64_t data_size = 6 * sizeof(float);
    std::vector<std::string> send_var_names{"w", "x"};
    int group_id = 0;
    int ret = heter_client_ptr_->Send(
        group_id, send_var_names, vars_len, values.data(), data_size);
    if (!ret) {
      LOG(INFO) << ">>>> TestShardSendRecv: worker send success";
    }
  };
  std::thread t(send_async);

  int group_id = 0;
  std::vector<std::string> recv_var_names{"w", "x"};
  int data_size = 6 * sizeof(float);
  float* value_ptr = new float[6];
  int ret =
      heter_client_ptr_->Recv(group_id, recv_var_names, value_ptr, data_size);
  if (!ret) {
    VLOG(4) << "queried data is: ";
    for (int i = 0; i < 6; i++) {
      VLOG(4) << value_ptr[i] << " ";
    }
    delete[] value_ptr;
    LOG(INFO) << "<<<< TestShardSendRecv: worker recv success";
  }

  t.join();
}

void PressTestSendRecv(
    std::shared_ptr<distributed::HeterClient> heter_client_ptr_) {
  // long l = 0, m = 0;
  // https://paddlerec.bj.bcebos.com/online_infer/arm_brpc_ubuntu18/send_20_34
  std::ifstream file("/send_20_34", std::ios::in | std::ios::binary);
  // l = file.tellg();
  // file.seekg(0, std::ios::end);
  // m = file.tellg();
  // file.close();
  // VLOG(0) << "size of file " << "20_34" << " is " << (m - l) << " bytes.\n";
  int64_t vars_len = 2359296 * sizeof(float);
  int64_t data_size = vars_len;
  VLOG(0) << "float num: " << data_size;
  float* data_ptr = new float[data_size];
  file.read(static_cast<char*>(data_ptr), 9437184);
  VLOG(0) << "send data is: " << data_ptr[0] << ", " << data_ptr[1];
  std::vector<std::string> var_names{"34"};
  int loopCnt = 10000;
  auto send_async = [&]() -> void {
    int i = 0;
    while (i++ < loopCnt) {
      heter_client_ptr_->Send(20, var_names, {vars_len}, data_ptr, data_size);
    }
  };
  std::thread t(send_async);
  float* values = new float[2359296];
  int i = 0;
  while (i++ < loopCnt) {
    int ret = heter_client_ptr_->Recv(20, var_names, values, data_size);
    if (!ret) {
      VLOG(0) << "diff: " << abs(values[0] - 0.159544) << ", "
              << abs(values[1] + 2.3484);
      VLOG(0) << "loop id: " << i;
      for (int j = 0; j < 2359296; j++) {
        if (abs(values[j] - data_ptr[j]) > 4e-6) {
          VLOG(0) << "error data idx: " << j;
          VLOG(0) << "diff detail: " << values[j] << ", " << data_ptr[j];
          LOG(INFO) << ">>>> worker recv ERROR";
          break;
        }
      }
      for (uint32_t i = 0; i < 2359296; i++) {
        values[i] = -1;  // reset
      }
    }
  }
  delete[] values;

  std::ofstream recv("/recv_20_34", std::ios::out | std::ios::binary);
  recv.write(static_cast<char*>(values, data_size));
  recv.close();
  t.join();
}

void TestScopeSendRecv(
    std::shared_ptr<distributed::HeterClient> heter_client_ptr_) {
  phi::CPUPlace place;
  phi::CPUContext ctx(place);
  framework::Executor exe(place);
  std::shared_ptr<framework::Scope> send_scope_ptr =
      std::make_shared<framework::Scope>();
  int64_t rows_numel = 10;
  InitTensorsOnClient(send_scope_ptr.get(), &place, rows_numel);
  LOG(INFO) << "InitTensorsOnClient done";
  auto send_async = [&]() -> void {
    std::string message_name = std::to_string(distributed::PS_SAVE_WITH_SCOPE);
    std::vector<std::string> send_var_names{"w", "x"};
    int ret = heter_client_ptr_->Send(
        ctx, *send_scope_ptr, message_name, send_var_names);
    if (!ret) {
      LOG(ERROR) << ">>>> TestScopeSendRecv: worker send success";
    }
  };
  std::thread t(send_async);

  std::string message_name = std::to_string(distributed::PS_QUERY_WITH_SCOPE);
  std::vector<std::string> recv_var_names{"w", "x"};
  std::shared_ptr<framework::Scope> recv_scope_ptr =
      std::make_shared<framework::Scope>();
  int ret = heter_client_ptr_->Recv(
      ctx, *recv_scope_ptr, message_name, recv_var_names);
  if (!ret && recv_scope_ptr->FindVar("w") && recv_scope_ptr->FindVar("x")) {
    LOG(INFO) << "<<<< TestScopeSendRecv: worker recv success";
  } else {
    LOG(INFO) << "<<<< TestScopeSendRecv: worker recv failed";
  }
  t.join();
}

TEST(HETERSENDANDRECV, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);

  // 启动 switch server A & B
  std::string switch_a_endpoint("127.0.0.1:6000");
  std::string switch_a_endpoint_inter("127.0.0.1:6100");
  std::string switch_b_endpoint_inter("127.0.0.1:7100");
  std::string switch_b_endpoint("127.0.0.1:7000");

  std::shared_ptr<distributed::HeterServer> switch_server_ptr_a =
      std::make_shared<distributed::HeterServer>();
  std::vector<std::string> end_points{switch_a_endpoint};
  std::vector<std::string> peer_endpoints{switch_b_endpoint_inter};
  std::thread switch_server_a_thread(StartSwitchServer,
                                     std::ref(switch_server_ptr_a),
                                     end_points,
                                     peer_endpoints);
  switch_server_ptr_a->WaitServerReady();

  std::shared_ptr<distributed::HeterServer> switch_server_ptr_b =
      std::make_shared<distributed::HeterServer>();
  end_points = {switch_b_endpoint, switch_b_endpoint_inter};
  peer_endpoints = {};
  std::thread switch_server_b_thread(StartSwitchServer,
                                     std::ref(switch_server_ptr_b),
                                     end_points,
                                     peer_endpoints);
  switch_server_ptr_b->WaitServerReady();

  end_points = {switch_b_endpoint, switch_b_endpoint_inter};
  peer_endpoints = {};
  std::thread switch_server_b_thread_inter(StartSwitchInterServer,
                                           std::ref(switch_server_ptr_b),
                                           end_points,
                                           peer_endpoints);
  switch_server_ptr_b->WaitServerReady();

  // 获取 client 实例
  // 开启单测时，请重新设置 HeterClient 端的 recv_switch_channels_
  std::shared_ptr<distributed::HeterClient> heter_client_ptr_ =
      distributed::HeterClient::GetInstance(
          {switch_a_endpoint, switch_b_endpoint}, {}, 0);

  framework::ProgramDesc program;
  phi::CPUPlace place;
  framework::Executor exe(place);
  exe.Prepare(program, 0);  // solve undefined symbol: tensor_table.cc

  // TestScopeSendRecv(heter_client_ptr_);
  // TestShardSendRecv(heter_client_ptr_);
  PressTestSendRecv(heter_client_ptr_);

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
