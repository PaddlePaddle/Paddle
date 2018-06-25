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

#pragma once

#include <time.h>

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "grpc++/channel.h"
#include "grpc++/generic/generic_stub.h"
#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "grpc/support/log.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace operators {
namespace distributed {

struct VarHandle {
  std::string ep;
  const platform::DeviceContext* ctx;
  const framework::Scope* scope;
  std::string name;

  std::string String() const {
    std::ostringstream s;
    s << "name:[" << name << "] ep:[" << ep << "]";
    return s.str();
  }
};

void ProcGetResponse(const VarHandle& var_h, const grpc::ByteBuffer& msg);

class BaseProcessor {
 public:
  explicit BaseProcessor(std::shared_ptr<grpc::Channel> ch) {
    context_ = nullptr;
  }

  virtual ~BaseProcessor() {}

  virtual void Prepare(const VarHandle& var_info, int64_t time_out) {
    context_.reset(new grpc::ClientContext());
    var_h_ = var_info;
    context_->set_wait_for_ready(true);

    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(time_out);

    context_->set_deadline(deadline);
  }

  virtual void Prepare(int64_t time_out) {
    context_.reset(new grpc::ClientContext());
    context_->set_wait_for_ready(true);

    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(time_out);

    context_->set_deadline(deadline);
  }

  virtual void Process() = 0;

  std::unique_ptr<grpc::ClientContext> context_;
  grpc::Status status_;
  VarHandle var_h_;
};

typedef std::function<void(const VarHandle&, const ::grpc::ByteBuffer&)>
    RequestSendCallBack;

class SendProcessor : public BaseProcessor {
 public:
  explicit SendProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor(ch), stub_g_(ch) {}

  virtual ~SendProcessor() {}

  virtual void Process() {
    if (response_call_back_) {
      response_call_back_(var_h_, reply_);
    }
  }

  ::grpc::GenericStub stub_g_;
  ::grpc::ByteBuffer reply_;
  RequestSendCallBack response_call_back_ = nullptr;
};

typedef std::function<void(const VarHandle&, const ::grpc::ByteBuffer&)>
    RequestGetCallBack;

class GetProcessor : public BaseProcessor {
 public:
  explicit GetProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor(ch), stub_g_(ch) {}

  virtual ~GetProcessor() {}

  virtual void Process() {
    if (response_call_back_) {
      response_call_back_(var_h_, reply_);
    }
  }

  ::grpc::ByteBuffer reply_;
  ::grpc::GenericStub stub_g_;
  RequestGetCallBack response_call_back_ = ProcGetResponse;
};

class BatchBarrierProcessor : public BaseProcessor {
 public:
  explicit BatchBarrierProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor(ch) {
    stub_ = sendrecv::SendRecvService::NewStub(ch);
  }

  virtual ~BatchBarrierProcessor() {}

  virtual void Process() {}
  sendrecv::VoidMessage reply_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
};

class FetchBarrierProcessor : public BaseProcessor {
 public:
  explicit FetchBarrierProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor(ch) {
    stub_ = sendrecv::SendRecvService::NewStub(ch);
  }

  virtual ~FetchBarrierProcessor() {}

  virtual void Process() {}
  sendrecv::VariableMessage reply_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
};

class GRPCClient : public RPCClient {
 public:
  GRPCClient() {}
  virtual ~GRPCClient();

  bool AsyncSendVar(const std::string& ep, const platform::DeviceContext& ctx,
                    const framework::Scope& scope, const std::string& var_name,
                    int64_t time_out = FLAGS_grpc_deadline) override;

  bool AsyncGetVar(const std::string& ep, const platform::DeviceContext& ctx,
                   const framework::Scope& scope, const std::string& var_name,
                   int64_t time_out = FLAGS_grpc_deadline) override;

  bool AsyncPrefetchVar(const std::string& ep,
                        const platform::DeviceContext& ctx,
                        const framework::Scope& scope,
                        const std::string& in_var_name,
                        const std::string& out_var_name,
                        int64_t time_out = FLAGS_grpc_deadline) override;

  void AsyncSendBatchBarrier(const std::string& ep,
                             int64_t time_out = FLAGS_grpc_deadline) override;

  void AsyncSendFetchBarrier(const std::string& ep,
                             int64_t time_out = FLAGS_grpc_deadline) override;

  void Wait() override;

  void SendComplete() override;

 protected:
  void InitImpl() override;

 private:
  // InitEventLoop should only be called by Init()
  void InitEventLoop();

  void Proceed();

  void AsyncSendComplete(const std::string& ep,
                         int64_t time_out = FLAGS_grpc_deadline);

  std::shared_ptr<grpc::Channel> GetChannel(const std::string& ep);

 private:
  grpc::CompletionQueue cq_;
  std::unordered_map<std::string, std::shared_ptr<grpc::Channel>> channels_;
  std::unique_ptr<std::thread> client_thread_;

  // mutex for Wait client sync
  std::mutex sync_mutex_;
  std::condition_variable sync_cond_;
  std::atomic<int64_t> req_count_{0};

  // mutex for GetChannel thread safety
  std::mutex chan_mutex_;
  DISABLE_COPY_AND_ASSIGN(GRPCClient);
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
