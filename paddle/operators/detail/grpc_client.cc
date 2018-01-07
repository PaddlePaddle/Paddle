/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
namespace paddle {
namespace operators {
namespace detail {

bool RPCClient::AsyncSendVariable(const std::string& ep,
                                  const platform::DeviceContext& ctx,
                                  const framework::Scope* scope,
                                  const std::string& var_name,
                                  int64_t time_out) {
  sendrecv::VariableMessage req;
  auto* var = scope->FindVar(var_name);
  SerializeToMessage(var_name, var, ctx, &req);

  // varhandle
  VarHandle var_h;
  var_h.ep = ep;
  var_h.scope = scope;
  var_h.name = var_name;
  var_h.ctx = &ctx;

  // stub context
  auto ch = GetChannel(ep);
  GRPCStubContext<sendrecv::VoidMessage>* c =
      new GRPCStubContext<sendrecv::VoidMessage>();
  c->init(ch, var_h, ProcSendResponse, time_out);

  // request context
  RequestContext* req_ctx = new RequestContext(kActionSend, c);

  req_contexts_[(void*)&req_ctx] = std::shared_ptr<RequestContext>(req_ctx);

  c->rpc = c->stub->AsyncSendVariable(c->context.get(), req, &cq_);
  c->rpc->Finish(c->reply.get(), &c->status, (void*)req_ctx);
  // int64_t a = 1;
  // c->rpc->Finish(c->reply.get(), &c->status, (void*)a);

  count_++;

  return true;
}

void ProcSendResponse(const VarHandle&, const sendrecv::VoidMessage& msg) {}

void ProcGetResponse(const VarHandle& var_h,
                     const sendrecv::VariableMessage& msg) {}

bool RPCClient::AsyncGetVariable(const std::string& ep,
                                 const platform::DeviceContext& ctx,
                                 const framework::Scope* scope,
                                 const std::string& var_name,
                                 int64_t time_out) {
  return true;
}

bool RPCClient::wait() {
  bool ok = true;

  while (true) {
    if (count_ <= 0) {
      break;
    }

    if (!Proceed()) {
      LOG(ERROR) << "Get meets CompletionQueue error";
      return false;
    }
  }

  return ok;
}

int RPCClient::ProcSendTag(RequestContext* req_context) {
  GRPCStubContext<sendrecv::VoidMessage>* c =
      (GRPCStubContext<sendrecv::VoidMessage>*)(req_context->ctx);
  if (!c->status.ok()) {
    RequestContext::destroy(req_context);
    req_contexts_.erase(req_context);
    return -1;
  }

  c->response_call_back(c->var_h, *c->reply.get());
  RequestContext::destroy(req_context);
  req_contexts_.erase(req_context);
  return 0;
}

int RPCClient::ProcGetTag(RequestContext* req_context) {
  auto c = (GRPCStubContext<sendrecv::VariableMessage>*)(req_context->ctx);
  if (!c->status.ok()) {
    RequestContext::destroy(req_context);
    req_contexts_.erase(req_context);
    return -1;
  }

  c->response_call_back(c->var_h, *c->reply.get());
  RequestContext::destroy(req_context);
  req_contexts_.erase(req_context);
  return 0;
}

bool RPCClient::Proceed() {
  void* tag = NULL;
  bool ok = false;

  // request counts.
  if (!cq_.Next(&tag, &ok)) {
    return false;
  }
  count_--;

  GPR_ASSERT(ok);
  PADDLE_ENFORCE(tag);

  // TODO(gongwb): add more retries.
  RequestContext* req_context = (RequestContext*)tag;
  switch (req_context->type) {
    case kActionSend: {
      ProcSendTag(req_context);
      break;
    }
    case kActionGet: {
      ProcGetTag(req_context);
      break;
    }
    default: { assert(false); }
  }

  return true;
}

std::shared_ptr<grpc::Channel> RPCClient::GetChannel(const std::string& ep) {
  auto it = channels_.find(ep);
  if (it != channels_.end()) {
    return it->second;
  }

  auto ch = std::shared_ptr<grpc::Channel>(
      grpc::CreateChannel(ep, grpc::InsecureChannelCredentials()));

  channels_[ep] = ch;
  return ch;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
