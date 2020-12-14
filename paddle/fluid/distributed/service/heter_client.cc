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

DEFINE_int32(pserver_timeout_ms, 10800000, "pserver request server timeout_ms");

std::shared_ptr<HeterClient> HeterClient::s_instance_ = NULL;
bool HeterClient::is_initialized_ = false;

void HeterClient::MainThread() {
  while (running_) {
    RpcProfilerControl();
  }
}

void HeterClient::Stop() {
  running_ = false;
  if (!is_initialized_) {
    VLOG(0) << "HeterClient is not inited, do nothing";
  } else {
    if (main_thread_) {
      auto status = StopHeterWorker();
      status.wait();
      main_thread_->join();
      main_thread_.reset(nullptr);
    }
    VLOG(1) << "HeterClient Stop Done";
  }
}

void HeterClient::RpcProfilerControl() {
  if (trainer_id_ == 0) {
    if (!do_server_profiler_ && platform::IsProfileEnabled()) {
      // send profiler start flag
      do_server_profiler_ = true;
      auto start_status = StartProfiler();
      start_status.wait();
    } else if (do_server_profiler_ && !platform::IsProfileEnabled()) {
      // send profiler end flag
      auto stop_status = StopProfiler();
      stop_status.wait();
      do_server_profiler_ = false;
    }
  }
}

void HeterClient::CreateClient2XpuConnection() {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "single";
  options.timeout_ms = pserver_timeout_ms;

  xpu_channels_.resize(xpu_list_.size());
  for (size_t i = 0; i < xpu_list_.size(); ++i) {
    xpu_channels_[i].reset(new brpc::Channel());
    if (xpu_channels_[i]->Init(xpu_list_[i].c_str(), "", &options) != 0) {
      VLOG(0) << "HeterServer channel init fail";
    }
  }
}

void HeterClient::SendAndRecvAsync(
    const std::vector<std::string>& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& message_name,
    const std::vector<std::string>& send_var_name,
    const std::vector<std::string>& recv_var_name) {
  platform::RecordEvent record_event("HeterClient->SendAndRecvAsync");
  const platform::DeviceContext* p_ctx = &ctx;
  const framework::Scope* p_scope = &scope;
  const std::string message_name_val = message_name;
  const std::vector<std::string> send_var_name_val = send_var_name;
  const std::vector<std::string> recv_var_name_val = recv_var_name;

  VLOG(3) << "GRPCClient::SendAndRecv Begin, message_name: "
          << message_name_val;
  // Todo: get correct channel
  int num = trainer_id_ % xpu_channels_.size();

  brpc::Controller cntl;
  cntl.set_timeout_ms(pserver_timeout_ms);
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

std::future<int32_t> HeterClient::SendCmd(
    uint32_t table_id, int cmd_id, const std::vector<std::string>& params) {
  size_t request_call_num = xpu_channels_.size();
  paddle::distributed::DownpourBrpcClosure* closure =
      new paddle::distributed::DownpourBrpcClosure(
          request_call_num, [request_call_num, cmd_id](void* done) {
            int ret = 0;
            auto* closure = (paddle::distributed::DownpourBrpcClosure*)done;
            for (size_t i = 0; i < request_call_num; ++i) {
              if (closure->check_response(i, cmd_id) != 0) {
                ret = -1;
                break;
              }
            }
            closure->set_promise_value(ret);
          });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(cmd_id);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(trainer_id_);
    for (const auto& param : params) {
      closure->request(i)->add_params(param);
    }
    ::paddle::PsService_Stub rpc_stub(xpu_channels_[i].get());
    closure->cntl(i)->set_timeout_ms(
        pserver_timeout_ms);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> HeterClient::StartProfiler() {
  return SendCmd(-1, PS_START_PROFILER, {});
}

std::future<int32_t> HeterClient::StopProfiler() {
  return SendCmd(-1, PS_STOP_PROFILER, {});
}

}  // end namespace distributed
}  // end namespace paddle
