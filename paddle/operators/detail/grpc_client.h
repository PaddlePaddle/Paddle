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

#pragma once

#include <time.h>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "paddle/framework/data_type.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/operators/detail/simple_block_queue.h"

#include "paddle/operators/detail/send_recv.grpc.pb.h"
#include "paddle/operators/detail/send_recv.pb.h"

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

namespace paddle {
namespace operators {
namespace detail {

// TODO(gongwb): add device support.
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

void SerializeToMessage(const std::string& name, const framework::Variable* var,
                        const platform::DeviceContext& ctx,
                        sendrecv::VariableMessage* msg);

void DeserializeFromMessage(const sendrecv::VariableMessage& msg,
                            const platform::DeviceContext& ctx,
                            framework::Variable* var);

void ProcSendResponse(const VarHandle&, const sendrecv::VoidMessage& msg);

void ProcGetResponse(const VarHandle& var_h,
                     const sendrecv::VariableMessage& msg);

template <typename ResponseT>
struct GRPCStubContext {
  void init(std::shared_ptr<grpc::Channel> ch, const VarHandle& var_info,
            std::function<void(const VarHandle&, const ResponseT&)> f,
            int64_t time_out) {
    stub = sendrecv::SendRecvService::NewStub(ch);
    context.reset(new grpc::ClientContext());
    status.reset(new grpc::Status());
    reply.reset(new ResponseT());
    var_h = var_info;

    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(time_out);

    context->set_deadline(deadline);
    response_call_back = f;
  }

  std::unique_ptr<sendrecv::SendRecvService::Stub> stub;
  std::unique_ptr<grpc::ClientContext> context;
  std::unique_ptr<grpc::Status> status;
  std::unique_ptr<ResponseT> reply;
  VarHandle var_h;

  std::function<void(const VarHandle&, const ResponseT&)> response_call_back;
};

enum ActionType {
  kActionSend = 0,
  kActionGet,
};

struct RequestContext {
  ActionType type;
  void* ctx;
};

typedef std::function<void(const VarHandle&, const sendrecv::VoidMessage&)>
    RequestSendCallBack;

typedef std::function<void(const VarHandle&, const sendrecv::VariableMessage&)>
    RequestGetCallBack;

class RPCClient {
 public:
  bool AsyncSendVariable(const std::string& ep,
                         const platform::DeviceContext& ctx,
                         const framework::Scope* scope,
                         const std::string& var_name,
                         int64_t time_out = 600 * 1000);

  bool AsyncGetVariable(const std::string& ep,
                        const platform::DeviceContext& ctx,
                        const framework::Scope* scope,
                        const std::string& var_name,
                        int64_t time_out = 600 * 1000);
  bool wait();

 private:
  bool Proceed();
  int ProcTag(RequestContext* tag);
  std::shared_ptr<grpc::Channel> GetChannel(const std::string& ep);

 private:
  grpc::CompletionQueue cq_;
  std::map<std::string, std::shared_ptr<grpc::Channel>> channels_;

  std::map<void*, std::shared_ptr<RequestContext>> req_contexts_;
  int64_t count_ = 0;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
