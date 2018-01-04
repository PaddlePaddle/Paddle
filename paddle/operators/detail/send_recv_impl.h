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

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/framework/var_type.h"
#include "paddle/operators/detail/simple_block_queue.h"

#include "paddle/operators/detail/send_recv.grpc.pb.h"
#include "paddle/operators/detail/send_recv.pb.h"

#include <grpc++/grpc++.h>

using grpc::Channel;
using grpc::Server;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerBuilder;

using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using sendrecv::SendRecvService;
using sendrecv::VariableMessage;
using sendrecv::VoidMessage;

namespace paddle {
namespace operators {
namespace detail {

typedef std::pair<std::string, sendrecv::VariableMessage> MessageWithName;

class SendRecvServerImpl final : public SendRecvService::Service {
 public:
  explicit SendRecvServerImpl() {}

  Status SendVariable(ServerContext *context, const VariableMessage *in_var,
                      VoidMessage *out_var) override;
  Status GetVariable(ServerContext *context, const VariableMessage *in_var,
                     VariableMessage *out_var) override;
  Status Wait(ServerContext *context, const VoidMessage *in_var,
              VoidMessage *out_var) override;
  void Reset();
  void Done();
  void SetScope(framework::Scope *scope) { scope_ = scope; };

  const MessageWithName Get() { return this->var_recv_queue_.Pop(); }

  void Push(const MessageWithName &msg) { this->var_recv_queue_.Push(msg); }

 private:
  // received variable from RPC, operators fetch variable from this queue.
  SimpleBlockQueue<MessageWithName> var_recv_queue_;
  framework::Scope *scope_;
  // condition of the sub program
  std::mutex mutex_;
  bool done_;
  std::condition_variable condition_;
};

// RPCClient is a class to send tensors to pserver sub-network
// using different hashing methods.
class RPCClient {
 public:
  RPCClient(std::shared_ptr<Channel> channel)
      : stub_(SendRecvService::NewStub(channel)) {}

  bool SendVariable(const framework::Scope &scope, const std::string &inname);
  bool GetVariable(const framework::Scope &scope, const std::string &outname);
  void Wait();

 private:
  std::unique_ptr<SendRecvService::Stub> stub_;
};

inline void SerializeToMessage(const std::string &name,
                               const framework::Variable *var,
                               const platform::DeviceContext &ctx,
                               VariableMessage *msg) {
  msg->set_varname(name);
  std::ostringstream oss;
  switch (framework::ToVarType(var->Type())) {
    case framework::proto::VarDesc_VarType_LOD_TENSOR: {
      msg->set_type(sendrecv::VarType::LOD_TENSOR);
      framework::SerializeToStream(oss, var->Get<framework::LoDTensor>(), ctx);
      break;
    }
    case framework::proto::VarDesc_VarType_SELECTED_ROWS: {
      msg->set_type(sendrecv::VarType::SELECTED_ROWS);
      framework::SerializeToStream(oss, var->Get<framework::SelectedRows>(),
                                   ctx);
      break;
    }
    default: {
      PADDLE_THROW("Serialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
    }
  }
  msg->set_serialized(oss.str());
}

inline void DeserializeFromMessage(const VariableMessage &msg,
                                   const platform::DeviceContext &ctx,
                                   framework::Variable *var) {
  using namespace paddle::framework::proto;
  std::istringstream iss(msg.serialized());
  switch (msg.type()) {
    case sendrecv::VarType::LOD_TENSOR: {
      framework::LoDTensor *tensor = var->GetMutable<framework::LoDTensor>();
      DeserializeFromStream(iss, tensor, ctx);
      break;
    }
    case sendrecv::VarType::SELECTED_ROWS: {
      framework::SelectedRows *rows =
          var->GetMutable<framework::SelectedRows>();
      DeserializeFromStream(iss, rows, ctx);
      break;
    }
    default: {
      PADDLE_THROW("Deserialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
    }
  }
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
