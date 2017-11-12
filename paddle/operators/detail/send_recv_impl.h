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
#include "paddle/operators/detail/simple_block_queue.h"

#include <grpc++/channel.h>
#include <grpc++/client_context.h>
#include <grpc++/create_channel.h>
#include <grpc++/security/credentials.h>
#include <grpc/grpc.h>
#include "paddle/operators/send_recv.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using sendrecv::SendRecvOp;
using sendrecv::SendTensor;

namespace paddle {
namespace operators {
namespace detail {

class SendRecvServerImpl final : public SendRecvOp::Service {
 public:
  explicit SendRecvServerImpl() {}

  void SetScope(framework::Scope *scope) { scope_ = scope; }

  Status InitVariables(ServerContext *context,
                       ServerReader<VariableMessage> *in_var_reader) override;

  Status SendTensor(ServerContext *context, const std::string *in_tensor,
                    std::string *out_tensor) override;

  const framework::LodTensor &Get() const { return lodtensor_queue_.Pop(); }

  void Push(framework::LodTensor &tensor) {
    lodtensor_return_queue_.Push(tensor);
  }

 private:
  framework::Scope *scope_;
  SimpleBlockQueue<framework::LodTensor> lodtensor_queue_;
  SimpleBlockQueue<framework::LodTensor> lodtensor_return_queue_;
  SimpleBlockQueue<framework::SelectedRows> selected_rows_queue_;
  SimpleBlockQueue<framework::SelectedRows> selected_rows_return_queue_;
};

// RPCClient is a class to send tensors to pserver sub-network
// using different hashing methods.
class RPCClient {
 public:
  RPCClient(std::shared_ptr<Channel> channel)
      : stub_(SendRecvOp::NewStub(channel)) {}

  bool SendTensor(const framework::LoDTensor &tensor);

 private:
  std::unique_ptr<SendRecvOp::Stub> stub_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
