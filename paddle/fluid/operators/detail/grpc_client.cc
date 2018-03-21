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
#include <sys/time.h>
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
    // sendrecv::VariableMessage req;
    // SerializeToMessage(var_name_val, var, *p_ctx, &req);

    /*
    struct timeval t0_wait, t1_wait;
    gettimeofday(&t0_wait, 0);
    std::thread::id this_id = std::this_thread::get_id();
    */

    ::grpc::ByteBuffer req;
    SerializeToByteBuffer(var_name_val, var, *p_ctx, &req);

    PrintDetail(var_name_val, var->Get<framework::LoDTensor>(), *p_ctx);
    /*
    gettimeofday(&t1_wait, 0);
    double t_wait = double((t1_wait.tv_sec - t0_wait.tv_sec) * 1000.0 +
                           (t1_wait.tv_usec - t0_wait.tv_usec) / 1000.0);
    auto tensor = var->Get<framework::LoDTensor>();
    std::stringstream ss;
    ss << "send var_name:" << var_name_val << ", dims: " << tensor.dims()
       << ", msg_len:" << req.Length() << ", time:" << t_wait
       << "ms, thread_id:" << this_id;
    std::cout << ss.str() << std::endl;
    */

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

    // auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
    // rpc->Finish(&s->reply_, &s->status_, (void*)s);

    auto call = std::move(s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/SendVariable", req,
        &cq_));
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, (void*)s);
  });

  req_count_++;

  return true;
}

void ProcGetResponse(const VarHandle& var_h,
                     // const sendrecv::VariableMessage& ret_msg) {
                     const ::grpc::ByteBuffer& ret_msg) {
  /*
struct timeval t0_wait, t1_wait;
gettimeofday(&t0_wait, 0);
*/

  // auto* outvar = var_h.scope->FindVar(var_h.name);
  framework::Variable* outvar = NULL;
  DeserializeFromByteBuffer(ret_msg, *var_h.ctx, var_h.scope, outvar);

  /*
  std::thread::id this_id = std::this_thread::get_id();
  gettimeofday(&t1_wait, 0);
  double t_wait = double((t1_wait.tv_sec - t0_wait.tv_sec) * 1000.0 +
                         (t1_wait.tv_usec - t0_wait.tv_usec) / 1000.0);

  // std::cout << "received:" << var_h.String() <<  ret_msg.Length() <<
  // std::endl;
  auto tensor = outvar->Get<framework::LoDTensor>();
  std::stringstream ss;
  ss << "received var_name:" << var_h.name << ", dims: " << tensor.dims()
     << ", msg_len:" << ret_msg.Length() << ", time:" << t_wait
     << "ms, thread_id:" << this_id;
  std::cout << ss.str() << '\n';
  */
}

template <typename T>
void RequestToByteBuffer(const T& proto, ::grpc::ByteBuffer* result) {
  ::grpc::Slice slice(proto.ByteSizeLong());
  proto.SerializeWithCachedSizesToArray(
      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(slice.begin())));
  ::grpc::ByteBuffer tmp(&slice, 1);
  result->Swap(&tmp);
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

    ::grpc::ByteBuffer buf;
    RequestToByteBuffer<sendrecv::VariableMessage>(req, &buf);

    // auto rpc = s->stub_->AsyncGetVariable(s->context_.get(), req, &cq_);
    // rpc->Finish(&s->reply_, &s->status_, (void*)s);
    auto call = std::move(s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/GetVariable", buf, &cq_));
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, (void*)s);
  });

  req_count_++;

  return true;
}

void RPCClient::AsyncSendBatchBarrier(const std::string& ep, int64_t time_out) {
  const auto ch = GetChannel(ep);

  BatchBarrierProcessor* s = new BatchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(BATCH_BARRIER_MESSAGE);
  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, (void*)s);
  req_count_++;
}

void RPCClient::AsyncSendFetchBarrier(const std::string& ep, int64_t time_out) {
  const auto ch = GetChannel(ep);
  FetchBarrierProcessor* s = new FetchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(FETCH_BARRIER_MESSAGE);
  auto rpc = s->stub_->AsyncGetVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, (void*)s);
  req_count_++;
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
  BaseProcessor* c = static_cast<BaseProcessor*>(tag);
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
  args.SetInt("grpc.testing.fixed_reconnect_backoff_ms", 5000);
  args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
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
