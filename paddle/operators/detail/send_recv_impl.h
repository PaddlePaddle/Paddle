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

#include "paddle/framework/data_type.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/selected_rows.h"
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

typedef std::pair<std::string, framework::LoDTensor> TensorWithName;

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

  const TensorWithName Get() { return this->var_recv_queue_.Pop(); }

 private:
  // received variable from RPC, operators fetch variable from this queue.
  SimpleBlockQueue<TensorWithName> var_recv_queue_;
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

}  // namespace detail
}  // namespace operators
}  // namespace paddle
