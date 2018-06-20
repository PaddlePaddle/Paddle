// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/distributed/brpc_client.h"
#include "paddle/fluid/framework/threadpool.h"

namespace paddle {
namespace operators {
namespace distributed {

DEFINE_int32(brpc_channel_num, 24,
             "Number of channels to send requests connected to one server");
DEFINE_int32(timeout_ms, 30000, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");

BRPCClient::~BRPCClient() { Wait(); }

void HandleSendResponse(brpc::Controller* cntl,
                        sendrecv::VoidMessage* response) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sendrecv::VoidMessage> response_guard(response);

  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send EchoRequest, " << cntl->ErrorText();
    return;
  }
  LOG(INFO) << "Received response from " << cntl->remote_side()
            << " latency=" << cntl->latency_us() << "us";
}

bool BRPCClient::AsyncSendVar(const std::string& ep,
                              const platform::DeviceContext& ctx,
                              const framework::Scope& scope,
                              const std::string& var_name, int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch_ptr = GetChannel(ep_val);

  framework::AsyncIO(
      [var_name_val, p_ctx, ep_val, p_scope, time_out, ch_ptr, this] {
        auto ch_ctx = ch_ptr->Pop();
        brpc::Controller* cntl = new brpc::Controller();
        sendrecv::VoidMessage* response = new sendrecv::VoidMessage();
        cntl->set_timeout_ms(time_out);

        google::protobuf::Closure* done =
            brpc::NewCallback(&HandleSendResponse, cntl, response);

        sendrecv::VariableMessage request;
        ch_ctx->stub->SendVariable(cntl, &request, response, done);
      });
  req_count_++;

  return true;
}

void HandleGetResponse(brpc::Controller* cntl,
                       sendrecv::VariableMessage* response) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sendrecv::VariableMessage> response_guard(response);

  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send EchoRequest, " << cntl->ErrorText();
    return;
  }
  LOG(INFO) << "Received response from " << cntl->remote_side()
            << " latency=" << cntl->latency_us() << "us";

  // framework::Variable* outvar = nullptr;
  // DeserializeFromByteBuffer(ret_msg, *var_h.ctx, var_h.scope, &outvar);
}

bool BRPCClient::AsyncGetVar(const std::string& ep,
                             const platform::DeviceContext& ctx,
                             const framework::Scope& scope,
                             const std::string& var_name, int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::AsyncIO(
      [var_name_val, ep_val, p_scope, p_ctx, time_out, ch, this] {});

  req_count_++;

  return true;
}

bool BRPCClient::AsyncPrefetchVar(const std::string& ep,
                                  const platform::DeviceContext& ctx,
                                  const framework::Scope& scope,
                                  const std::string& in_var_name,
                                  const std::string& out_var_name,
                                  int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string in_var_name_val = in_var_name;
  const std::string out_var_name_val = out_var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::AsyncIO([in_var_name_val, out_var_name_val, ep_val, p_scope, p_ctx,
                      time_out, ch, this] {});

  req_count_++;
  return true;
}

void BRPCClient::AsyncSendBatchBarrier(const std::string& ep,
                                       int64_t time_out) {
  req_count_++;
}

void BRPCClient::AsyncSendFetchBarrier(const std::string& ep,
                                       int64_t time_out) {
  req_count_++;
}

void BRPCClient::Wait() {
  std::unique_lock<std::mutex> lk(sync_mutex_);
  sync_cond_.wait(lk, [this] { return req_count_ == 0; });
}

ChannelQueuePtr BRPCClient::GetChannel(const std::string& ep) {
  {
    std::lock_guard<std::mutex> guard(chan_mutex_);
    auto it = channels_.find(ep);
    if (it != channels_.end()) {
      return it->second;
    }
  }

  ChannelQueuePtr q(new framework::BlockingQueue<ChannelContextPtr>());

  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "pooled";
  options.connect_timeout_ms = 100;
  options.timeout_ms = FLAGS_timeout_ms /*milliseconds*/;
  options.max_retry = FLAGS_max_retry;
  for (int i = 0; i < FLAGS_brpc_channel_num; ++i) {
    std::shared_ptr<ChannelContext> c(new ChannelContext());
    if (c->channel.Init(ep.c_str(), &options) != 0) {
      LOG(ERROR) << "Fail to initialize channel";
      return nullptr;
    }

    c->stub.reset(new sendrecv::SendRecvService_Stub(
        static_cast<google::protobuf::RpcChannel*>(&c->channel)));
    q->Push(c);
  }

  {
    std::lock_guard<std::mutex> guard(chan_mutex_);
    channels_[ep] = q;
  }

  return q;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
