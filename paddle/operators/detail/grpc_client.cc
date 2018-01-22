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
                                  const framework::Scope& scope,
                                  const std::string& var_name,
                                  int64_t time_out) {
  sendrecv::VariableMessage req;
  auto* var = scope.FindVar(var_name);
  SerializeToMessage(var_name, var, ctx, &req);

  // varhandle
  VarHandle var_h;
  var_h.ep = ep;
  var_h.scope = &scope;
  var_h.name = var_name;
  var_h.ctx = &ctx;

  // stub context
  auto ch = GetChannel(ep);
  SendProcessor* s = new SendProcessor(ch);
  s->Prepare(var_h, time_out);
  s->response_call_back_ = NULL;

  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, (void*)s);

  req_count_++;

  return true;
}

void ProcGetResponse(const VarHandle& var_h,
                     const sendrecv::VariableMessage& ret_msg) {
  auto* outvar = var_h.scope->FindVar(var_h.name);

  std::istringstream iss(ret_msg.serialized());
  DeserializeFromMessage(ret_msg, *var_h.ctx, outvar);
}

bool RPCClient::AsyncGetVariable(const std::string& ep,
                                 const platform::DeviceContext& ctx,
                                 const framework::Scope& scope,
                                 const std::string& var_name,
                                 int64_t time_out) {
  sendrecv::VariableMessage req;
  req.set_varname(var_name);

  // varhandle
  VarHandle var_h;
  var_h.ep = ep;
  var_h.scope = &scope;
  var_h.name = var_name;
  var_h.ctx = &ctx;

  // stub context
  auto ch = GetChannel(ep);
  GetProcessor* s = new GetProcessor(ch);
  s->Prepare(var_h, time_out);
  s->response_call_back_ = ProcGetResponse;

  auto rpc = s->stub_->AsyncGetVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, (void*)s);

  req_count_++;

  return true;
}

bool RPCClient::Wait() {
  bool ok = true;

  while (true) {
    if (req_count_ <= 0) {
      break;
    }

    if (!Proceed()) {
      return false;
    }
  }

  return ok;
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
  req_count_--;
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

  auto ch = std::shared_ptr<grpc::Channel>(
      grpc::CreateCustomChannel(ep, grpc::InsecureChannelCredentials(), args));

  channels_[ep] = ch;
  return ch;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
