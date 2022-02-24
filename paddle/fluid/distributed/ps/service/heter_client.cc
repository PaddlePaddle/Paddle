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

#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/split.h"

DECLARE_int32(rpc_deadline);
DECLARE_int32(pserver_timeout_ms);

namespace paddle {
namespace distributed {

std::shared_ptr<HeterClient> HeterClient::s_instance_ = NULL;
bool HeterClient::is_initialized_ = false;

int GetMicroId(const platform::DeviceContext& ctx,
               const framework::Scope* scope) {
  framework::Variable* var = scope->FindVar("microbatch_id");
  PADDLE_ENFORCE_EQ(var->IsType<framework::LoDTensor>(), true,
                    platform::errors::InvalidArgument(
                        "the type of micro id shoulde be LoDTensor."));
  auto micro_id = -1;
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  if (platform::is_cpu_place(tensor->place())) {
    auto data = reinterpret_cast<const float*>(tensor->data());
    micro_id = static_cast<int>(data[0]);
  } else {
#ifdef PADDLE_WITH_CUDA
    std::vector<char> temp;
    temp.resize(tensor->numel() * framework::DataTypeSize(tensor->dtype()));
    char* temp_ptr = temp.data();
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(
        platform::CPUPlace(), temp_ptr, tensor->place(), tensor->data(),
        tensor->numel() * framework::DataTypeSize(tensor->dtype()), stream);
    float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
    micro_id = static_cast<int>(temp_ptr_float[0]);
#endif
  }
  return micro_id;
}

void HeterClient::MainThread() {
  while (running_) {
    RpcProfilerControl();
  }
}

void HeterClient::Stop() {
  running_ = false;
  if (!is_initialized_) {
    VLOG(3) << "HeterClient is not inited, do nothing";
  } else {
    if (main_thread_) {
      auto status = StopHeterWorker();
      status.wait();
      main_thread_->join();
      main_thread_.reset(nullptr);
    }
    VLOG(3) << "HeterClient Stop Done";
  }
}

void HeterClient::FinalizeWorker() {
  running_ = false;
  if (!is_initialized_) {
    VLOG(3) << "HeterClient is not inited, do nothing";
  } else {
    if (main_thread_) {
      main_thread_->join();
      main_thread_.reset(nullptr);
    }
    VLOG(3) << "HeterClient Stop Done";
  }
}

std::future<int32_t> HeterClient::StopHeterWorker() {
  return SendCmd(-1, PS_STOP_SERVER, {});
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
  options.timeout_ms = FLAGS_pserver_timeout_ms;

  xpu_channels_.resize(xpu_list_.size());
  for (size_t i = 0; i < xpu_list_.size(); ++i) {
    xpu_channels_[i].reset(new brpc::Channel());
    if (xpu_channels_[i]->Init(xpu_list_[i].c_str(), "", &options) != 0) {
      VLOG(0) << "HeterClient channel init fail. Try Again";
      auto ip_port = paddle::string::Split(xpu_list_[i], ':');
      std::string ip = ip_port[0];
      int port = std::stoi(ip_port[1]);
      std::string int_ip_port = GetIntTypeEndpoint(ip, port);
      if (xpu_channels_[i]->Init(int_ip_port.c_str(), "", &options) != 0) {
        LOG(ERROR) << "BrpcPsServer start failed, ip_port= " << int_ip_port;
      }
    }
  }
  previous_xpu_channels_.resize(previous_xpu_list_.size());
  for (size_t i = 0; i < previous_xpu_list_.size(); ++i) {
    previous_xpu_channels_[i].reset(new brpc::Channel());
    if (previous_xpu_channels_[i]->Init(previous_xpu_list_[i].c_str(), "",
                                        &options) != 0) {
      VLOG(0) << "HeterClient channel init fail. Try Again";
      auto ip_port = paddle::string::Split(previous_xpu_list_[i], ':');
      std::string ip = ip_port[0];
      int port = std::stoi(ip_port[1]);
      std::string int_ip_port = GetIntTypeEndpoint(ip, port);
      if (previous_xpu_channels_[i]->Init(int_ip_port.c_str(), "", &options) !=
          0) {
        LOG(ERROR) << "BrpcPsServer start failed, ip_port= " << int_ip_port;
      }
    }
  }
}

void HeterClient::SendAndRecvAsync(
    const platform::DeviceContext& ctx, const framework::Scope& scope,
    const std::string& message_name,
    const std::vector<std::string>& send_var_name,
    const std::vector<std::string>& recv_var_name, const std::string& mode) {
  platform::RecordEvent record_event("HeterClient->SendAndRecvAsync",
                                     platform::TracerEventType::Communication,
                                     1);
  const platform::DeviceContext* p_ctx = &ctx;
  const framework::Scope* p_scope = &scope;
  const std::string message_name_val = message_name;
  const std::vector<std::string> send_var_name_val = send_var_name;
  const std::vector<std::string> recv_var_name_val = recv_var_name;
  VLOG(3) << "BRPCClient::SendAndRecv Begin, message_name: "
          << message_name_val;
  brpc::Channel* channel = nullptr;
  distributed::MultiVarMsg request;
  OnHeterRpcDone* closure = new OnHeterRpcDone([p_ctx, p_scope](void* done) {
    auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
    PADDLE_ENFORCE_NE(
        closure->cntl.Failed(), true,
        platform::errors::Unimplemented(
            "HeterClient::SendAndRecv meets brpc error, error message is %s",
            closure->cntl.ErrorText()));

    VLOG(4) << "call heter_worker success";
  });
  closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);
  auto& request_io_buffer = closure->cntl.request_attachment();
  distributed::SerializeToMultiVarMsgAndIOBuf(
      message_name_val, send_var_name_val, recv_var_name_val, *p_ctx, p_scope,
      &request, &request_io_buffer);

  int micro_id = GetMicroId(ctx, p_scope);
  auto minibatch_id = micro_id / 10;
  // select channel according to micro id
  if (mode == "forward") {
    int num = minibatch_id % xpu_channels_.size();
    channel = xpu_channels_[num].get();
  } else if (mode == "backward") {
    int num = minibatch_id % previous_xpu_channels_.size();
    channel = previous_xpu_channels_[num].get();
  }
  ::paddle::distributed::PsService_Stub stub(channel);
  stub.SendAndRecvVariable(&closure->cntl, &request, &closure->response,
                           closure);
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
    ::paddle::distributed::PsService_Stub rpc_stub(xpu_channels_[i].get());
    closure->cntl(i)->set_timeout_ms(
        FLAGS_pserver_timeout_ms);  // cmd msg don't limit timeout for save/load
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
