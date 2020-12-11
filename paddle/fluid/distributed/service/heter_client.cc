// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/distributed/service/heter_client.h"
#include <algorithm>
#include <utility>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/timer.h"

DECLARE_int32(rpc_deadline);
namespace paddle {
namespace distributed {

std::shared_ptr<HeterClient> HeterClient::s_instance_ = NULL;
bool HeterClient::is_initialized_ = false;

void HeterClient::CreateClient2XpuConnection() {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "single";
  options.timeout_ms = 2000000;

  xpu_channels_.resize(xpu_list_.size());
  for (size_t i = 0; i < xpu_list_.size(); ++i) {
    xpu_channels_[i].reset(new brpc::Channel());
    if (xpu_channels_[i]->Init(xpu_list_[i].c_str(), "", &options) != 0) {
      VLOG(0) << "HeterServer channel init fail";
    }
  }
}

void HeterClient::SetXpuList(const std::vector<std::string>& xpu_list) {
  for (auto& x : xpu_list) {
    xpu_list_.push_back(x);
  }
}

void HeterClient::SendAndRecvAsync(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& message_name,
    const std::vector<std::string>& send_var_name,
    const std::vector<std::string>& recv_var_name) {
  platform::RecordEvent record_event("HeterClient->SendAndRecvAsync");
  const std::string ep_val = ep;
  const platform::DeviceContext* p_ctx = &ctx;
  const framework::Scope* p_scope = &scope;
  const std::string message_name_val = message_name;
  const std::vector<std::string> send_var_name_val = send_var_name;
  const std::vector<std::string> recv_var_name_val = recv_var_name;

  VLOG(3) << "GRPCClient::SendAndRecv Begin, message_name: "
          << message_name_val;
  // Todo: get correct channel
  int num = 0;

  brpc::Controller cntl;
  cntl.set_timeout_ms(10800000);
  distributed::MultiVarMsg request, response;
  auto& request_io_buffer = cntl.request_attachment();
  ::paddle::PsService_Stub stub(xpu_channels_[num].get());
  distributed::SerializeToMultiVarMsgAndIOBuf(
      message_name_val, send_var_name_val, recv_var_name_val, *p_ctx, p_scope,
      &request, &request_io_buffer);
  stub.SendAndRecvVariable(&cntl, &request, &response, NULL);
  PADDLE_ENFORCE_NE(
      cntl.Failed(), true,
      platform::errors::Unimplemented(
          "HeterClient::SendAndRecv meets brpc error, error message is %s",
          cntl.ErrorText()));
  VLOG(4) << "call heter_worker success";
  auto& response_io_buffer = cntl.response_attachment();
  distributed::DeserializeFromMultiVarMsgAndIOBuf(response, &response_io_buffer,
                                                  ctx, p_scope);
}

}  // end namespace distributed
}  // end namespace paddle
