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

#include "paddle/fluid/operators/detail/grpc_client.h"
#include <grpc++/support/slice.h>

#include <sys/time.h>
#include <limits>

#include "paddle/fluid/framework/threadpool.h"

namespace paddle {
namespace operators {
namespace detail {

bool RPCClient::AsyncSendVariable(const std::string &ep,
                                  const platform::DeviceContext &ctx,
                                  const framework::Scope &scope,
                                  const std::string &var_name,
                                  int64_t time_out) {
  const platform::DeviceContext *p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope *p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::AsyncIO([var_name_val, p_ctx, ep_val, p_scope, time_out, ch,
                      this] {
    auto *var = p_scope->FindVar(var_name_val);

    ::grpc::ByteBuffer req;
    SerializeToByteBuffer(var_name_val, var, *p_ctx, &req);

    // varhandle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = var_name_val;
    var_h.ctx = p_ctx;

    // stub context
    SendProcessor *s = new SendProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = NULL;

    auto call = s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/SendVariable", req, &cq_);
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, reinterpret_cast<void *>(s));
  });

  req_count_++;

  return true;
}

void ProcGetResponse(const VarHandle &var_h,
                     const ::grpc::ByteBuffer &ret_msg) {
  framework::Variable *outvar = nullptr;
  DeserializeFromByteBuffer(ret_msg, *var_h.ctx, var_h.scope, &outvar);
}

template <typename T>
void RequestToByteBuffer(const T &proto, ::grpc::ByteBuffer *result) {
  ::grpc::Slice slice(proto.ByteSizeLong());
  proto.SerializeWithCachedSizesToArray(const_cast<uint8_t *>(slice.begin()));
  ::grpc::ByteBuffer tmp(&slice, 1);
  result->Swap(&tmp);
}

bool RPCClient::AsyncGetVariable(const std::string &ep,
                                 const platform::DeviceContext &ctx,
                                 const framework::Scope &scope,
                                 const std::string &var_name,
                                 int64_t time_out) {
  const platform::DeviceContext *p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope *p_scope = &scope;
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
    GetProcessor *s = new GetProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = ProcGetResponse;

    auto call = s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/GetVariable", buf, &cq_);
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, reinterpret_cast<void *>(s));
  });

  req_count_++;
  return true;
}

bool RPCClient::AsyncPrefetchVariable(const std::string &ep,
                                      const platform::DeviceContext &ctx,
                                      const framework::Scope &scope,
                                      const std::string &in_var_name,
                                      const std::string &out_var_name,
                                      int64_t time_out) {
  const platform::DeviceContext *p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string in_var_name_val = in_var_name;
  const std::string out_var_name_val = out_var_name;
  const framework::Scope *p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::AsyncIO([in_var_name_val, out_var_name_val, ep_val, p_scope, p_ctx,
                      time_out, ch, this] {
    auto *var = p_scope->FindVar(in_var_name_val);

    ::grpc::ByteBuffer req;
    SerializeToByteBuffer(in_var_name_val, var, *p_ctx, &req, out_var_name_val);

    // var handle
    VarHandle var_h;
    var_h.ep = ep_val;
    var_h.scope = p_scope;
    var_h.name = out_var_name_val;
    var_h.ctx = p_ctx;

    // stub context
    GetProcessor *s = new GetProcessor(ch);
    s->Prepare(var_h, time_out);
    s->response_call_back_ = ProcGetResponse;

    auto call = s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/PrefetchVariable", req,
        &cq_);
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, static_cast<void *>(s));
  });

  req_count_++;
  return true;
}

void RPCClient::AsyncSendBatchBarrier(const std::string &ep, int64_t time_out) {
  const auto ch = GetChannel(ep);

  BatchBarrierProcessor *s = new BatchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(BATCH_BARRIER_MESSAGE);
  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void *>(s));
  req_count_++;
}

void RPCClient::AsyncSendFetchBarrier(const std::string &ep, int64_t time_out) {
  const auto ch = GetChannel(ep);
  FetchBarrierProcessor *s = new FetchBarrierProcessor(ch);
  s->Prepare(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(FETCH_BARRIER_MESSAGE);
  auto rpc = s->stub_->AsyncGetVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void *>(s));
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
    waits[i] = framework::AsyncIO([i, &a, this] { a[i] = Proceed(); });
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
  void *tag = NULL;
  bool ok = false;

  // request counts.
  if (!cq_.Next(&tag, &ok)) {
    LOG(ERROR) << "Get meets CompletionQueue error";
    return false;
  }

  GPR_ASSERT(ok);
  PADDLE_ENFORCE(tag);

  // TODO(gongwb): add more retries.
  BaseProcessor *c = static_cast<BaseProcessor *>(tag);
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

std::shared_ptr<grpc::Channel> RPCClient::GetChannel(const std::string &ep) {
  auto it = channels_.find(ep);
  if (it != channels_.end()) {
    return it->second;
  }

  grpc::ChannelArguments args;
  args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
  args.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  args.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());

  auto ch =
      grpc::CreateCustomChannel(ep, grpc::InsecureChannelCredentials(), args);

  channels_[ep] = ch;
  return ch;
}

/******************************* MPI *************************************/

bool RPCClient::AsyncMPISendVariable(const std::string &ep,
                                     const platform::DeviceContext &ctx,
                                     const framework::Scope &scope,
                                     const std::string &var_name,
                                     int64_t time_out = 600 * 1000, int mpi_src,
                                     int mpi_tag) {
  const platform::DeviceContext *p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope *p_scope = &scope;
  const auto ch = GetChannel(ep_val);
  const int mpi_src = mpi_src;
  const int mpi_tag = mpi_tag;

  framework::AsyncIO(
      [var_name_val, p_ctx, ep_val, p_scope, time_out, ch, this] {
        auto *var = p_scope->FindVar(var_name_val);
        ::grpc::ByteBuffer var_req;
        SerializeToByteBuffer(var_name_val, var, *p_ctx, &var_req);
        std::vector<Slice> *slices = var_req.slices();

        std::string var_slices;
        for (auto const &slice : slices) {
          int size = static_cast<int>(slice.size());
          var_slices.append<int>(size);
          var_slices.append(":");
        }

        std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
        stub_ = sendrecv::SendRecvService::NewStub(ch);

        MPIVarStatus mpi_var_status;
        mpi_var_status.name = var_name_val;
        mpi_var_status.grpc = false;
        mpi_var_status.mpi = false;

        ClientContext context;
        VariableMeta grpc_req;
        grpc_req.var_name = var_name_val;

        grpc_req.var_length = var_req.Length();
        grpc_req.mpi_src = mpi_src;
        grpc_req.var_slices = var_slices;
        grpc_req.mpi_dst = -1;
        grpc_req.mpi_tag = mpi_tag;

        VariableMeta grpc_res;

        ::grpc::Status status =
            stub_.SendVariableMeta(&context, &grpc_req, *grpc_res);

        // todo(tangwei) push var send error to queue, break this async
        if (!status.ok()) {
          LOG(ERROR) << "name:[" << var_name_val << "] "
                     << "send SendVariableMeta error" << std::endl;
          mpi_cq_.push(mpi_var_status);
          return;
        }
        mpi_var_status.grpc = true;

        // todo(tangwei) judgement status is ok or not
        LOG(DEBUG) << "name:[" << var_name_val << "] src:[" << grpc_req.src()
                   << "] dst:[" << grpc_res.dst() << "]";
        // todo(tangwei) push mpi send status and result to queue

        std::vector<std::future<void>> fs;

        for (Slice slice : slices) {
          fs.push_back(framework::Async([&var_req, &grpc_res]() {
            MPI_Request request;
            MPI_Status status;
            MPIIsendProcess(*var_req, var_req.Length(), grpc_res.dst(),
                            grpc_req.tag(), &request, &status);
          }));
        }

        for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
        mpi_var_status.mpi = true;
        mpi_cq_.push(mpi_var_status);
      });

  req_count_++;
  return true;
}

bool RPCClient::AsyncMPIRecvVariable(const std::string &ep,
                                     const platform::DeviceContext &ctx,
                                     const framework::Scope &scope,
                                     const std::string &var_name,
                                     int64_t time_out = 600 * 1000) {
  const platform::DeviceContext *p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope *p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  framework::AsyncIO(
      [var_name_val, ep_val, p_scope, p_ctx, time_out, ch, this] {
        std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
        stub_ = sendrecv::SendRecvService::NewStub(ch);

        MPIVarStatus mpi_var_status;
        mpi_var_status.name = var_name_val;
        mpi_var_status.grpc = false;
        mpi_var_status.mpi = false;

        ClientContext context;
        VariableMeta var_req;
        VariableMeta var_res;

        ::grpc::Status status =
            stub_.SendVariableMeta(&context, &var_req, *var_res);

        // todo(tangwei) push var send error to queue, break this async
        if (!status.ok()) {
          LOG(ERROR) << "name:[" << var_name_val << "] "
                     << "send SendVariableMeta error" << std::endl;
          mpi_cq_.push(mpi_var_status);
          return;
        }
        mpi_var_status.grpc = true;

        // todo(tangwei) judgement status is ok or not
        LOG(DEBUG) << "name:[" << var_name_val << "] src:[" << var_req.src()
                   << "] dst:[" << var_res.dst() << "]";
        // todo(tangwei) push mpi send status and result to queue

        MPI_Request request;
        MPI_Status status;
        ::grpc::ByteBuffer recv_var;
        MPIIrecvProcess(*recv_var, var_res.length(), var_res.dst(),
                        var_req.tag(), &request, &status);

        mpi_var_status.mpi = true;
        mpi_cq_.push(mpi_var_status);
      });

  req_count_++;
  return true;
}

bool RPCClient::ProceedMPI() {
  MPIVarStatus *mpi_status;
  // request counts.
  if (!mpi_cq_.Receive(&mpi_status)) {
    LOG(ERROR) << "Get meets Mpi Complete Blocking Queue Error";
    return false;
  }
  if (!mpi_statis.mpi) {
    LOG(ERROR) << "proc param error:" << mpi_statis.name;
    return false;
  }
  return true;
}

bool RPCClient::WaitMPI() {
  if (req_count_ <= 0) {
    return true;
  }
  const size_t kReqCnt = req_count_;
  bool a[kReqCnt];
  std::vector<std::future<void>> waits(req_count_);

  for (int i = 0; i < req_count_; i++) {
    waits[i] = framework::AsyncIO([i, &a, this] { a[i] = ProceedMPI(); });
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
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
