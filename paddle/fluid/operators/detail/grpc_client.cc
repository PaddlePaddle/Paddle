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

#include "grpc_client.h"
#include "paddle/fluid/framework/threadpool.h"
namespace paddle {
namespace operators {
namespace detail {

bool RPCClient::AsyncSendVariable(const std::string& ep,
                                  const platform::DeviceContext& ctx,
                                  const framework::Scope& scope,
                                  const std::string& var_name,
                                  int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::Async([var_name_val, p_ctx, ep_val, p_scope, time_out, ch, this] {
    auto* var = p_scope->FindVar(var_name_val);
    sendrecv::VariableMessage req;
    SerializeToMessage(var_name_val, var, *p_ctx, &req);

    // varhandle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = var_name_val;
    var_h.ctx = p_ctx;

    // stub context
    SendProcessor* s = new SendProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = NULL;

    auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
    rpc->Finish(&s->reply_, &s->status_, (void*)s);
  });

  req_count_++;

  return true;
}

void ProcGetResponse(const VarHandle& var_h,
                     const sendrecv::VariableMessage& ret_msg) {
  auto* outvar = var_h.scope->FindVar(var_h.name);
  DeserializeFromMessage(ret_msg, *var_h.ctx, outvar);
}

bool RPCClient::AsyncGetVariable(const std::string& ep,
                                 const platform::DeviceContext& ctx,
                                 const framework::Scope& scope,
                                 const std::string& var_name,
                                 int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::Async([var_name_val, ep_val, p_scope, p_ctx, time_out, ch, this] {
    sendrecv::VariableMessage req;
    req.set_varname(var_name_val);

    // varhandle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = var_name_val;
    var_h.ctx = p_ctx;

    // stub context
    GetProcessor* s = new GetProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = ProcGetResponse;

    auto rpc = s->stub_->AsyncGetVariable(s->context_.get(), req, &cq_);
    rpc->Finish(&s->reply_, &s->status_, (void*)s);
  });

  req_count_++;

  return true;
}

bool RPCClient::AsyncSendBatchBarrier(const std::string& ep, int64_t time_out) {
  const auto ch = GetChannel(ep);

  BatchBarrierProcessor* s = new BatchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(BATCH_BARRIER_MESSAGE);
  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, (void*)s);
  req_count_++;

  return true;
}

bool RPCClient::Wait() {
  if (req_count_ <= 0) {
    return true;
  }
  const size_t kReqCnt = req_count_;
  bool a[kReqCnt];
  std::vector<std::future<void>> waits(req_count_);

  for (int i = 0; i < req_count_; i++) {
    waits[i] = framework::Async([i, &a, this] { a[i] = Proceed(); });
  }

  for (int i = 0; i < req_count_; i++) {
    waits[i].wait();
  }

  int last_req_count = req_count_;
  req_count_ = 0;

  for (int i = 0; i < last_req_count; i++) {
    if (!a[i]) {
      return false;
    }
  }

  return true;
}

bool RPCClient::Proceed() {
  void* tag = NULL;
  bool ok = false;

  // request counts.
  if (!cq_.Next(&tag, &ok)) {
    LOG(ERROR) << "Get meets CompletionQueue error";
    return false;
  }

  GPR_ASSERT(ok);
  PADDLE_ENFORCE(tag);

  // TODO(gongwb): add more retries.
  ClientBase* c = static_cast<ClientBase*>(tag);
  if (!c->status_.ok()) {
    LOG(ERROR) << "proc param error:" << c->var_h_.String()
               << " grpc error:" << c->status_.error_message();
    delete c;
    return false;
  }

  c->Process();
  delete c;
  return true;
}

std::shared_ptr<grpc::Channel> RPCClient::GetChannel(const std::string& ep) {
  auto it = channels_.find(ep);
  if (it != channels_.end()) {
    return it->second;
  }

  grpc::ChannelArguments args;
  args.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  args.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());

  auto ch =
      grpc::CreateCustomChannel(ep, grpc::InsecureChannelCredentials(), args);

  channels_[ep] = ch;
  return ch;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
