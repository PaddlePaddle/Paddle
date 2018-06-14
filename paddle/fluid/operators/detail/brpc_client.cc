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

#include "paddle/fluid/operators/detail/brpc_client.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/detail/brpc_sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace detail {

DEFINE_int32(brpc_channel_num, 24,
             "Number of channels to send requests connected to one server");
DEFINE_int32(timeout_ms, 30000, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");

BRPCClient::~BRPCClient() { Wait(); }

void HandleSendResponse(brpc::Controller* cntl, sendrecv::VoidMessage* response,
                        VarHandle var_h, ChannelQueuePtr ch_ptr,
                        ChannelContextPtr ch_ctx, BRPCClient* cls) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sendrecv::VoidMessage> response_guard(response);

  // this channel can be used by other now.
  ch_ptr->Push(ch_ctx);
  cls->DecreaseReqCount();

  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send SendVar: " << var_h.name
                 << ", error text: " << cntl->ErrorText();
    return;
  }

  VLOG(4) << "Received SendResponse from: " << cntl->remote_side()
          << ", varname: " << var_h.name << ", latency: " << cntl->latency_us()
          << "us";
  VLOG(4) << "Finish HandleSendResponse";
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

        auto* var = p_scope->FindVar(var_name_val);
        sendrecv::VariableMessage request;
        SerializeToIOBuf(var_name_val, var, *p_ctx, &request,
                         &cntl->request_attachment());

        // varhandle
        VarHandle var_h;
        var_h.ep = ep_val;
        var_h.scope = p_scope;
        var_h.name = var_name_val;
        var_h.ctx = p_ctx;

        google::protobuf::Closure* done = brpc::NewCallback(
            &HandleSendResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

        ch_ctx->stub->SendVariable(cntl, &request, response, done);
      });
  req_count_++;

  return true;
}

void HandleGetResponse(brpc::Controller* cntl,
                       sendrecv::VariableMessage* response, VarHandle var_h,
                       ChannelQueuePtr ch_ptr, ChannelContextPtr ch_ctx,
                       BRPCClient* cls) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sendrecv::VariableMessage> response_guard(response);

  // this channel can be used other now.
  ch_ptr->Push(ch_ctx);

  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send SendVar: " << var_h.name
                 << ", error text: " << cntl->ErrorText();
    cls->DecreaseReqCount();
    return;
  }

  VLOG(4) << "Received SendResponse from: " << cntl->remote_side()
          << ", varname: " << var_h.name << ", latency: " << cntl->latency_us()
          << "us";

  framework::Variable* outvar = nullptr;
  detail::DeserializeFromIOBuf(*response, cntl->response_attachment(),
                               *var_h.ctx, var_h.scope, &outvar);
  VLOG(4) << "Finish HandleGetResponse";
  cls->DecreaseReqCount();
}

bool BRPCClient::AsyncGetVar(const std::string& ep,
                             const platform::DeviceContext& ctx,
                             const framework::Scope& scope,
                             const std::string& var_name, int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch_ptr = GetChannel(ep_val);

  framework::AsyncIO(
      [var_name_val, ep_val, p_scope, p_ctx, time_out, ch_ptr, this] {
        auto ch_ctx = ch_ptr->Pop();

        brpc::Controller* cntl = new brpc::Controller();
        sendrecv::VariableMessage* response = new sendrecv::VariableMessage();
        cntl->set_timeout_ms(time_out);

        sendrecv::VariableMessage req;
        req.set_varname(var_name_val);

        // var handle
        VarHandle var_h;
        var_h.ep = ep_val;
        var_h.scope = p_scope;
        var_h.name = var_name_val;
        var_h.ctx = p_ctx;

        google::protobuf::Closure* done = brpc::NewCallback(
            &HandleGetResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

        ch_ctx->stub->GetVariable(cntl, &req, response, done);
      });

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
  const auto ch_ptr = GetChannel(ep_val);

  framework::AsyncIO([in_var_name_val, out_var_name_val, ep_val, p_scope, p_ctx,
                      time_out, ch_ptr, this] {
    auto ch_ctx = ch_ptr->Pop();

    brpc::Controller* cntl = new brpc::Controller();
    sendrecv::VariableMessage* response = new sendrecv::VariableMessage();
    cntl->set_timeout_ms(time_out);

    auto* var = p_scope->FindVar(in_var_name_val);
    sendrecv::VariableMessage req;
    detail::SerializeToIOBuf(in_var_name_val, var, *p_ctx, &req,
                             &cntl->request_attachment(), out_var_name_val);

    // var handle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = out_var_name_val;
    var_h.ctx = p_ctx;

    google::protobuf::Closure* done = brpc::NewCallback(
        &HandleGetResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

    ch_ctx->stub->PrefetchVariable(cntl, &req, response, done);
  });

  req_count_++;
  return true;
}

void BRPCClient::AsyncSendBatchBarrier(const std::string& ep,
                                       int64_t time_out) {
  AsyncSendMessage(ep, BATCH_BARRIER_MESSAGE, time_out);
}

void BRPCClient::AsyncSendFetchBarrier(const std::string& ep,
                                       int64_t time_out) {
  AsyncSendMessage(ep, FETCH_BARRIER_MESSAGE, time_out);
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

void BRPCClient::SendComplete() {
  for (auto& it : channels_) {
    this->AsyncSendComplete(it.first);
  }
}

void BRPCClient::AsyncSendComplete(const std::string& ep, int64_t time_out) {
  AsyncSendMessage(ep, COMPLETE_MESSAGE, time_out);
}

void BRPCClient::AsyncSendMessage(const std::string& ep,
                                  const std::string& message,
                                  int64_t time_out) {
  auto ch_ptr = GetChannel(ep);
  auto ch_ctx = ch_ptr->Pop();

  brpc::Controller* cntl = new brpc::Controller();
  sendrecv::VoidMessage* response = new sendrecv::VoidMessage();
  cntl->set_timeout_ms(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(message);

  // varhandle
  VarHandle var_h;
  var_h.name = message;

  google::protobuf::Closure* done = brpc::NewCallback(
      &HandleSendResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

  ch_ctx->stub->SendVariable(cntl, &req, response, done);
  req_count_++;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
