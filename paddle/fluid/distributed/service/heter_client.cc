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
    VLOG(3) << "channel init: " << xpu_list_[i];
    xpu_channels_[i].reset(new brpc::Channel());
    if (xpu_channels_[i]->Init(xpu_list_[i].c_str(), "", &options) != 0) {
      VLOG(0) << "server channel init fail";
    }
  }
}

void HeterClient::SetXpuList(const std::vector<std::string>& xpu_list) {
  VLOG(3) << "Going to set xpu list";
  for (auto& x : xpu_list) {
    xpu_list_.push_back(x);
    VLOG(3) << "set xpu list:  " << x << " size: " << xpu_list_.size();
  }
}

void HeterClient::SendAndRecvAsync(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& message_name,
    const std::vector<std::string>& send_var_name,
    const std::vector<std::string>& recv_var_name) {
  const std::string ep_val = ep;
  const platform::DeviceContext* p_ctx = &ctx;
  const framework::Scope* p_scope = &scope;
  const std::string message_name_val = message_name;
  const std::vector<std::string> send_var_name_val = send_var_name;
  const std::vector<std::string> recv_var_name_val = recv_var_name;

  // distributed::SerializeToMultiVarMsg(message_name_val, send_var_name_val,
  //                       recv_var_name_val, *p_ctx, p_scope, &request);

  VLOG(1) << "GRPCClient::SendAndRecv Begin, message_name: "
          << message_name_val;
  // Todo: get correct channel
  int num = 0;

  // distributed::OnHeterRpcDone* done = new OnHeterRpcDone([&](void* done) {
  //   auto* closure = reinterpret_cast<distributed::OnHeterRpcDone*>(done);
  //   if (closure->cntl.Failed()) {
  //     VLOG(1) << "call heter_worker fail: " << closure->cntl.ErrorText();
  //   } else {
  //     VLOG(1) << "call heter_worker success! Server endpoint "<<
  //     closure->cntl.remote_side(); auto &response_io_buffer =
  //     closure->cntl.response_attachment();
  //     distributed::DeserializeFromMultiVarMsgAndIOBuf(closure->response,
  //     &response_io_buffer, ctx, p_scope);
  //   }
  // });

  // VLOG(1) << "GRPCClient::SendAndRecv Get PsService_Stub";
  // ::paddle::PsService_Stub stub(xpu_channels_[num].get());
  // VLOG(1) << "GRPCClient::SendAndRecv SendAndRecvVariable";
  // auto cid = &done->cntl.call_id();
  // distributed::MultiVarMsg request;
  // auto &request_io_buffer = &done->cntl.request_attachment();
  // distributed::SerializeToMultiVarMsgAndIOBuf(message_name_val,
  // send_var_name_val, recv_var_name_val, *p_ctx, p_scope, &request,
  // &request_io_buffer);

  // done->cntl.set_timeout_ms(FLAGS_rpc_deadline);
  // stub.SendAndRecvVariable(&done->cntl, &request, &done->response, done);
  // brpc::Join(cid);
  VLOG(1) << "GRPCClient::SendAndRecv SendAndRecvVariable Finish";

  brpc::Controller cntl;
  cntl.set_timeout_ms(10800000);
  distributed::MultiVarMsg request, response;
  auto& request_io_buffer = cntl.request_attachment();
  ::paddle::PsService_Stub stub(xpu_channels_[num].get());
  distributed::SerializeToMultiVarMsgAndIOBuf(
      message_name_val, send_var_name_val, recv_var_name_val, *p_ctx, p_scope,
      &request, &request_io_buffer);
  stub.SendAndRecvVariable(&cntl, &request, &response, NULL);
  if (cntl.Failed()) {
    VLOG(1) << "call heter_worker fail: " << cntl.ErrorText();
  } else {
    VLOG(1) << "call heter_worker success";
    auto& response_io_buffer = cntl.response_attachment();
    distributed::DeserializeFromMultiVarMsgAndIOBuf(
        response, &response_io_buffer, ctx, p_scope);
  }
}

}  // end namespace distributed
}  // end namespace paddle
