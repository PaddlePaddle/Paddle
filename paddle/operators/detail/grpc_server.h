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
#include <grpc/support/log.h>

using grpc::Channel;
using grpc::Server;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
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

class AsyncGRPCServer final : public SendRecvService::Service {
 public:
  explicit AsyncGRPCServer(std::string address) { address_ = address; }

  void Run();

  /*
  Status SendVariable(ServerContext *context, const VariableMessage *in_var,
                      VoidMessage *out_var) override;

  Status GetVariable(ServerContext *context, const VariableMessage *in_var,
                     VariableMessage *out_var) override;

  Status Wait(ServerContext *context, const VoidMessage *in_var,
              VoidMessage *out_var) override;

  void Reset();
  void Done();
  */
  void SetScope(framework::Scope *scope) { scope_ = scope; }

  const TensorWithName Get() { return this->var_recv_queue_.Pop(); }

  void Push(const TensorWithName &msg) { this->var_recv_queue_.Push(msg); }

 protected:
  void HandleRpcs();

 private:
  SimpleBlockQueue<TensorWithName> var_recv_queue_;
  std::unique_ptr<ServerCompletionQueue> cq_;
  SendRecvService::AsyncService service_;
  std::unique_ptr<Server> server_;
  std::string address_;
  framework::Scope *scope_;
};

};  // namespace detail
};  // namespace operators
};  // namespace paddle
