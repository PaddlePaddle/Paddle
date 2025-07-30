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
#include "paddle/phi/core/platform/profiler.h"

namespace paddle::distributed {
PD_DEFINE_int32(heter_world_size, 100, "group size");  // group max size
PD_DEFINE_int32(switch_send_recv_timeout_s, 600, "switch_send_recv_timeout_s");

std::shared_ptr<HeterClient> HeterClient::s_instance_ = nullptr;
std::mutex HeterClient::mtx_;
std::shared_ptr<HeterClient> HeterClient::switch_s_instance_ = nullptr;

int GetMicroId(const phi::DeviceContext& ctx, const framework::Scope* scope) {
  framework::Variable* var = scope->FindVar("microbatch_id");
  PADDLE_ENFORCE_EQ(var->IsType<phi::DenseTensor>(),
                    true,
                    common::errors::InvalidArgument(
                        "the type of micro id should be DenseTensor."));
  auto micro_id = -1;
  auto* tensor = var->GetMutable<phi::DenseTensor>();
  if (phi::is_cpu_place(tensor->place())) {
    auto data = reinterpret_cast<const float*>(tensor->data());
    micro_id = static_cast<int>(data[0]);
  } else {
#ifdef PADDLE_WITH_CUDA
    std::vector<char> temp;
    temp.resize(tensor->numel() * phi::SizeOf(tensor->dtype()));
    char* temp_ptr = temp.data();
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(phi::CPUPlace(),
                 temp_ptr,
                 tensor->place(),
                 tensor->data(),
                 tensor->numel() * phi::SizeOf(tensor->dtype()),
                 stream);
    float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
    micro_id = static_cast<int>(temp_ptr_float[0]);
#endif
  }
  return micro_id;
}

void HeterClient::Stop() {
  auto status = StopHeterWorker();
  status.wait();
}

std::future<int32_t> HeterClient::StopHeterWorker() {
  return SendCmd(-1, PS_STOP_SERVER, {});
}

std::future<int32_t> HeterClient::StartProfiler() {
  return SendCmd(-1, PS_START_PROFILER, {});
}

std::future<int32_t> HeterClient::StopProfiler() {
  return SendCmd(-1, PS_STOP_PROFILER, {});
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
    if (previous_xpu_channels_[i]->Init(
            previous_xpu_list_[i].c_str(), "", &options) != 0) {
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
    const phi::DeviceContext& ctx,
    const framework::Scope& scope,
    const std::string& message_name,
    const std::vector<std::string>& send_var_name,
    const std::vector<std::string>& recv_var_name,
    const std::string& mode) {
  phi::RecordEvent record_event(
      "HeterClient->SendAndRecvAsync", phi::TracerEventType::Communication, 1);
  const phi::DeviceContext* p_ctx = &ctx;
  const framework::Scope* p_scope = &scope;
  const std::vector<std::string> send_var_name_val = send_var_name;
  const std::vector<std::string> recv_var_name_val = recv_var_name;
  VLOG(3) << "BRPCClient::SendAndRecv Begin, message_name: " << message_name;
  brpc::Channel* channel = nullptr;
  distributed::MultiVarMsg request;
  OnHeterRpcDone* closure = new OnHeterRpcDone([](void* done) {
    auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
    PADDLE_ENFORCE_NE(
        closure->cntl.Failed(),
        true,
        common::errors::Unimplemented(
            "HeterClient::SendAndRecv meets brpc error, error message is %s",
            closure->cntl.ErrorText()));
    VLOG(4) << "call heter_worker success";
  });
  closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);
  auto& request_io_buffer = closure->cntl.request_attachment();
  distributed::SerializeToMultiVarMsgAndIOBuf(message_name,
                                              send_var_name_val,
                                              recv_var_name_val,
                                              *p_ctx,
                                              p_scope,
                                              &request,
                                              &request_io_buffer);

  int micro_id = GetMicroId(ctx, p_scope);  // global
  auto minibatch_id = micro_id / 10;
  VLOG(4) << "micro_id: " << micro_id;
  // select channel according to micro id
  if (mode == "forward") {
    int num = minibatch_id % xpu_channels_.size();
    channel = xpu_channels_[num].get();
  } else if (mode == "backward") {
    int num = minibatch_id % previous_xpu_channels_.size();
    channel = previous_xpu_channels_[num].get();
  } else if (mode == "send_to_switch") {
    VLOG(4) << "calling switch service";
    // auto promise = std::make_shared<std::promise<int32_t>>();
    // closure->add_promise(promise);
    // std::future<int> fut = promise->get_future();
    // int idx = 1;  // for test
    // LOG(INFO) << "xpu_channels_ size: " << xpu_channels_.size();
    // channel = xpu_channels_[idx].get();  // 为了适配 send_and_recv op
    // paddle::distributed::PsService_Stub stub(channel);
    // stub.SendToSwitch(&closure->cntl, &request, &closure->response,
    // closure); fut.wait();
    VLOG(4) << "calling switch service done";
    return;
  }
  paddle::distributed::PsService_Stub stub(channel);
  stub.SendAndRecvVariable(
      &closure->cntl, &request, &closure->response, closure);
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
    paddle::distributed::PsService_Stub rpc_stub(xpu_channels_[i].get());
    closure->cntl(i)->set_timeout_ms(
        FLAGS_pserver_timeout_ms);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(
        closure->cntl(i), closure->request(i), closure->response(i), closure);
  }
  return fut;
}

int HeterClient::Send(const phi::DeviceContext& ctx,
                      const framework::Scope& scope,
                      const std::string& message_name,
                      const std::vector<std::string>& send_var_names) {
  const framework::Scope* p_scope = &scope;  // 注意是 const
  OnHeterRpcDone* closure = new OnHeterRpcDone([](void* done) {
    auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
    int ret = 0;
    closure->set_promise_value(ret);
    if (closure->cntl.Failed()) {
      PADDLE_ENFORCE_NE(
          closure->cntl.Failed(),
          true,
          common::errors::Unimplemented(
              "HeterClient::SendToSwitch meets brpc error, error message is %s",
              closure->cntl.ErrorText()));
    }
  });

  closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);
  auto& request_io_buffer = closure->cntl.request_attachment();

  distributed::MultiVarMsg request;
  // 1. set req message_name(string)
  request.set_message_name(message_name);
  request.set_group_id(0);

  // 2. set req send_var_names(<string>)
  for (auto& send_var_name : send_var_names) {
    request.add_send_var_names(send_var_name);
  }

  // 3. set req var_messages(<VarMessage>)
  for (auto& send_var_name : send_var_names) {
    auto* send_var_msg = request.add_var_messages();
    send_var_msg->set_varname(send_var_name);
    framework::Variable* var = p_scope->FindVar(send_var_name);
    butil::IOBuf temp_iobuf;
    if (var->IsType<phi::DenseTensor>()) {
      SerializeDenseTensor(var, ctx, send_var_msg, &temp_iobuf);
    } else if (var->IsType<phi::SelectedRows>()) {
      SerializeSelectedRows(var, ctx, send_var_msg, &temp_iobuf);
    }
    request_io_buffer.append(temp_iobuf);
  }
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  if (send_switch_channels_.empty()) {
    LOG(ERROR) << "send_switch_channels_ is null, get xpu_channels_[0]";
    if (xpu_channels_.empty()) {
      LOG(ERROR) << "xpu_channels_ is null";
    }
    send_switch_channels_.push_back(xpu_channels_[0]);
  }
  brpc::Channel* channel = send_switch_channels_[0].get();
  // brpc::Channel* channel = xpu_channels_[0].get();
  paddle::distributed::PsService_Stub stub(channel);
  stub.SendToSwitch(&closure->cntl, &request, &closure->ps_response, closure);

  VLOG(4) << "waiting SendToSwitch response result......";
  fut.wait();
  VLOG(4) << "Send done";
  return 0;
}

int HeterClient::Send(int group_id,
                      const std::vector<std::string>& var_names,
                      const std::vector<int64_t>& vars_size,
                      void* data_ptr,
                      int64_t data_size) {
  OnHeterRpcDone* closure = new OnHeterRpcDone([](void* done) {
    auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
    int ret = 0;
    closure->set_promise_value(ret);
    if (closure->cntl.Failed()) {
      LOG(ERROR) << "Send meets brpc error, err msg is %s"
                 << closure->cntl.ErrorText();
    }
  });
  distributed::MultiVarMsg request;
  closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);
  std::string message_name = "send and save";
  request.set_message_name(message_name);
  request.set_group_id(group_id);
  for (auto& send_var_name : var_names) {
    request.add_send_var_names(send_var_name);
  }
  for (auto var_len : vars_size) {
    request.add_vars_len(var_len);
  }
  auto& request_buffer = closure->cntl.request_attachment();
  request_buffer.append(reinterpret_cast<void*>(data_ptr), data_size);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  if (send_switch_channels_.empty()) {
    LOG(ERROR) << "send_switch_channels_ is null, get xpu_channels_[0]";
    if (xpu_channels_.empty()) {
      LOG(ERROR) << "xpu_channels_ is null";
    }
    send_switch_channels_.push_back(xpu_channels_[0]);
  }
  brpc::Channel* channel = send_switch_channels_[0].get();
  paddle::distributed::PsService_Stub stub(channel);
  stub.SendToSwitch(&closure->cntl, &request, &closure->ps_response, closure);
  fut.wait();
  delete closure;
  return 0;
}

int HeterClient::Recv(const phi::DeviceContext& ctx,
                      framework::Scope& recv_scope,  // NOLINT
                      const std::string& message_name,
                      const std::vector<std::string>& recv_var_names) {
  OnHeterRpcDone* closure = new OnHeterRpcDone([](void* done) {
    auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
    VLOG(4) << "Recv service call done";
    int ret = 0;
    closure->set_promise_value(ret);
    if (closure->cntl.Failed()) {
      VLOG(4) << "HeterClient::RecvFromSwitch meets "
                 "brpc error, error message is %s"
              << closure->cntl.ErrorText();
    }
  });

  closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);

  distributed::MultiVarMsg request;
  // 1. set req message_name(string)
  request.set_message_name(message_name);
  request.set_group_id(0);

  // 2. set req recv_var_names(<string>)
  for (auto& recv_var_name : recv_var_names) {
    request.add_recv_var_names(recv_var_name);
  }
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  if (recv_switch_channels_.empty()) {
    LOG(ERROR) << "peer_switch_channels_ is null, get xpu_channels_[1]";
    if (xpu_channels_.size() < 2) {
      LOG(ERROR) << "xpu_channels_ is null";
    }
    recv_switch_channels_.push_back(xpu_channels_[1]);
  }
  brpc::Channel* channel = recv_switch_channels_[0].get();
  paddle::distributed::PsService_Stub stub(channel);
  stub.RecvFromSwitch(&closure->cntl, &request, &closure->response, closure);
  fut.wait();
  VLOG(4) << "RecvFromSwitch done";
  // save in worker
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  phi::CPUPlace cpu_place;
  auto& cpu_dev_ctx = *pool.Get(cpu_place);
  auto& res_io_buffer = closure->cntl.response_attachment();
  VLOG(4) << "entering DeserializeFromMultiVarMsgAndIOBuf";
  distributed::DeserializeFromMultiVarMsgAndIOBuf(
      closure->response, &res_io_buffer, cpu_dev_ctx, &recv_scope);
  VLOG(4) << "Recv done";
  return 0;
}

int HeterClient::Recv(int group_id,
                      const std::vector<std::string>& var_names,
                      void* data_ptr,
                      int64_t data_size) {
  OnHeterRpcDone* closure = new OnHeterRpcDone([](void* done) {
    auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
    int ret = 0;
    closure->set_promise_value(ret);
    if (closure->cntl.Failed()) {
      LOG(ERROR) << "Recv meets brpc error, err msg is %s"
                 << closure->cntl.ErrorText();
    }
  });
  closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);

  distributed::MultiVarMsg request;
  std::string message_name = "query and recv";
  request.set_message_name(message_name);
  request.set_group_id(group_id);

  for (auto& recv_var_name : var_names) {
    request.add_recv_var_names(recv_var_name);
  }
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  if (recv_switch_channels_.empty()) {
    LOG(ERROR) << "peer_switch_channels_ is null, get xpu_channels_[1]";
    if (xpu_channels_.size() < 2) {
      LOG(ERROR) << "xpu_channels_ is null";
    }
    recv_switch_channels_.push_back(xpu_channels_[0]);
  }
  brpc::Channel* channel = recv_switch_channels_[0].get();
  paddle::distributed::PsService_Stub stub(channel);
  stub.RecvFromSwitch(&closure->cntl, &request, &closure->response, closure);
  fut.wait();
  VLOG(4) << "RecvFromSwitch done";
  // save in worker
  auto& res_io_buffer = closure->cntl.response_attachment();
  butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
  io_buffer_itr.copy_and_forward(reinterpret_cast<void*>(data_ptr), data_size);
  delete closure;
  VLOG(4) << "Recv done";
  return 0;
}
}  // namespace paddle::distributed
