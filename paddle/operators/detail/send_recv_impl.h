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

// #include <grpc++/channel.h>
// #include <grpc++/client_context.h>
// #include <grpc++/create_channel.h>
// #include <grpc++/security/credentials.h>
#include "paddle/operators/detail/send_recv.grpc.pb.h"
#include "paddle/operators/detail/send_recv.pb.h"

#include <grpc++/grpc++.h>

using grpc::Channel;
using grpc::ServerContext;
using grpc::ServerReader;

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

class SendRecvServerImpl final : public SendRecvService::Service {
 public:
  explicit SendRecvServerImpl() {}

  void SetScope(framework::Scope *scope) { scope_ = scope; }

  Status InitVariables(ServerContext *context,
                       ServerReader<VariableMessage> *in_var_reader,
                       VoidMessage *void_ret) override;

  Status SendVariable(ServerContext *context, const VariableMessage *in_var,
                      VariableMessage *out_var) override;

  const framework::LoDTensor Get() { return this->lodtensor_queue_.Pop(); }

  void Push(const framework::LoDTensor &tensor) {
    this->lodtensor_return_queue_.Push(tensor);
  }

 private:
  // Scope for send recv to run.
  framework::Scope *scope_;
  SimpleBlockQueue<framework::LoDTensor> lodtensor_queue_;
  SimpleBlockQueue<framework::LoDTensor> lodtensor_return_queue_;
  SimpleBlockQueue<framework::SelectedRows> selected_rows_queue_;
  SimpleBlockQueue<framework::SelectedRows> selected_rows_return_queue_;
};

// RPCClient is a class to send tensors to pserver sub-network
// using different hashing methods.
class RPCClient {
 public:
  RPCClient(std::shared_ptr<Channel> channel)
      : stub_(SendRecvService::NewStub(channel)) {}

  void SetScope(framework::Scope *scope) { scope_ = scope; }
  bool InitVariables(const std::vector<std::string> &var_list);
  bool SendVariable(const std::string &inname, const std::string &outname);

 private:
  std::unique_ptr<SendRecvService::Stub> stub_;
  // FIXME(typhoonzero): borrow scope pointer, this is not thread-safe!
  framework::Scope *scope_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
