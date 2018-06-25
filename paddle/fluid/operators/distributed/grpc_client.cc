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

#include "paddle/fluid/operators/distributed/grpc_client.h"

#include <sys/time.h>

#include <limits>

#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace distributed {

void GRPCClient::InitImpl() { InitEventLoop(); }

void GRPCClient::InitEventLoop() {
  // start the client process thread
  // TODO(wuyi): can make this in a threadpool
  client_thread_.reset(new std::thread(std::bind(&GRPCClient::Proceed, this)));
}

void GRPCClient::SendComplete() {
  for (auto& it : channels_) {
    this->AsyncSendComplete(it.first);
  }
}

GRPCClient::~GRPCClient() {
  Wait();
  cq_.Shutdown();
  {
    std::lock_guard<std::mutex> guard(chan_mutex_);
    for (auto& it : channels_) {
      it.second.reset();
    }
  }
  client_thread_->join();
}

bool GRPCClient::AsyncSendVar(const std::string& ep,
                              const platform::DeviceContext& ctx,
                              const framework::Scope& scope,
                              const std::string& var_name, int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::AsyncIO([var_name_val, p_ctx, ep_val, p_scope, time_out, ch,
                      this] {
    auto* var = p_scope->FindVar(var_name_val);

    ::grpc::ByteBuffer req;
    SerializeToByteBuffer(var_name_val, var, *p_ctx, &req);

    // varhandle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = var_name_val;
    var_h.ctx = p_ctx;

    // stub context
    SendProcessor* s = new SendProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = nullptr;

    auto call = s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/SendVariable", req, &cq_);
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  });
  req_count_++;

  return true;
}

void ProcGetResponse(const VarHandle& var_h,
                     const ::grpc::ByteBuffer& ret_msg) {
  framework::Variable* outvar = nullptr;
  DeserializeFromByteBuffer(ret_msg, *var_h.ctx, var_h.scope, &outvar);
}

template <typename T>
void RequestToByteBuffer(const T& proto, ::grpc::ByteBuffer* result) {
  ::grpc::Slice slice(proto.ByteSizeLong());
  proto.SerializeWithCachedSizesToArray(const_cast<uint8_t*>(slice.begin()));
  ::grpc::ByteBuffer tmp(&slice, 1);
  result->Swap(&tmp);
}

bool GRPCClient::AsyncGetVar(const std::string& ep,
                             const platform::DeviceContext& ctx,
                             const framework::Scope& scope,
                             const std::string& var_name, int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::AsyncIO([var_name_val, ep_val, p_scope, p_ctx, time_out, ch,
                      this] {
    // prepare input
    sendrecv::VariableMessage req;
    req.set_varname(var_name_val);
    ::grpc::ByteBuffer buf;
    RequestToByteBuffer<sendrecv::VariableMessage>(req, &buf);

    // var handle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = var_name_val;
    var_h.ctx = p_ctx;

    // stub context
    GetProcessor* s = new GetProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = ProcGetResponse;

    auto call = s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/GetVariable", buf, &cq_);
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  });

  req_count_++;

  return true;
}

bool GRPCClient::AsyncPrefetchVar(const std::string& ep,
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
                      time_out, ch, this] {
    auto* var = p_scope->FindVar(in_var_name_val);

    ::grpc::ByteBuffer req;
    SerializeToByteBuffer(in_var_name_val, var, *p_ctx, &req, out_var_name_val);

    // var handle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = out_var_name_val;
    var_h.ctx = p_ctx;

    // stub context
    GetProcessor* s = new GetProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = ProcGetResponse;

    auto call = s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/PrefetchVariable", req,
        &cq_);
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, static_cast<void*>(s));
  });

  req_count_++;
  return true;
}

void GRPCClient::AsyncSendBatchBarrier(const std::string& ep,
                                       int64_t time_out) {
  const auto ch = GetChannel(ep);

  BatchBarrierProcessor* s = new BatchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(BATCH_BARRIER_MESSAGE);
  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;
}

void GRPCClient::AsyncSendFetchBarrier(const std::string& ep,
                                       int64_t time_out) {
  const auto ch = GetChannel(ep);
  FetchBarrierProcessor* s = new FetchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(FETCH_BARRIER_MESSAGE);
  auto rpc = s->stub_->AsyncGetVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;
}

void GRPCClient::AsyncSendComplete(const std::string& ep, int64_t time_out) {
  const auto ch = GetChannel(ep);

  BatchBarrierProcessor* s = new BatchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(COMPLETE_MESSAGE);
  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;
}

void GRPCClient::Wait() {
  std::unique_lock<std::mutex> lk(sync_mutex_);
  sync_cond_.wait(lk, [this] { return req_count_ == 0; });
}

void GRPCClient::Proceed() {
  void* tag = nullptr;
  bool ok = false;

  while (cq_.Next(&tag, &ok)) {
    BaseProcessor* c = static_cast<BaseProcessor*>(tag);
    GPR_ASSERT(ok);
    PADDLE_ENFORCE(c);
    if (c->status_.ok()) {
      c->Process();
    } else {
      LOG(FATAL) << "var: " << c->var_h_.String()
                 << " grpc error:" << c->status_.error_message();
    }
    delete c;
    {
      std::lock_guard<std::mutex> lk(sync_mutex_);
      req_count_--;
    }
    sync_cond_.notify_all();
  }
}

std::shared_ptr<grpc::Channel> GRPCClient::GetChannel(const std::string& ep) {
  std::lock_guard<std::mutex> guard(chan_mutex_);
  auto it = channels_.find(ep);
  if (it != channels_.end()) {
    return it->second;
  }

  // Channel configurations:
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 2000);
  args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
  args.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  args.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());

  auto ch =
      grpc::CreateCustomChannel(ep, grpc::InsecureChannelCredentials(), args);
  channels_[ep] = ch;
  return ch;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
