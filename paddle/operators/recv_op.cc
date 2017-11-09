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

#include <stdint.h>
#include <sys/stat.h>
#include <ostream>
#include <thread>

#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/detail/simple_block_queue.h"

#include <grpc++/security/server_credentials.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/server_context.h>
#include <grpc/grpc.h>
#include "paddle/operators/send_recv.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
using sendrecv::SendRecvOp;
using sendrecv::TensorMessage;

class SendRecvServerImpl final : public SendRecvOp::Service {
 public:
  explicit SendRecvServerImpl() {}

  Status SendTensor(ServerContext *context, const std::string *in_tensor,
                    std::string *out_tensor) override {
    framework::LodTensor t;
    // TODO(typhoonzero): desirealize in_tensor and run pserver network.
    std::istringstream iss(*in_tensor);
    framework::Tensor t;
    framework::DesirializeFromStream(iss, &t);
    lodtensor_queue_.Push(std::move(t));
    // Block util the sub graph is done.
    auto t = lodtensor_return_queue_.Pop();
    std::ostringstream oss;
    framework::SerializeToStream(oss, &t);
    *out_tensor = oss.str();
  }

  Status SendTensorStream(
      ServerContext *context,
      ServerReaderWriter<TensorMessage, TensorMessage> *stream) override {
    // TODO(typhoonzero): implement stream methods.
    return Status::OK;
  }

  Status SendSelectedRows(ServerContext *context, const std::string *in_sr,
                          std::string *out_sr) {
    // TODO(typhoonzero): implement SendSelectedRows
    return Status::OK;
  }

  Status SendSelectedRowsStream(
      ServerContext *context,
      ServerReaderWriter<std::string, std::string> *stream) override {
    // TODO(typhoonzero): implement SendSelectedRowsStream
    return Status::OK;
  }

  const framework::LodTensor &Get() const { return lodtensor_queue_.Pop(); }

  void Push(framework::LodTensor &tensor) {
    lodtensor_return_queue_.Push(tensor);
  }

 private:
  SimpleBlockQueue<framework::LodTensor> lodtensor_queue_;
  SimpleBlockQueue<framework::LodTensor> lodtensor_return_queue_;
  SimpleBlockQueue<framework::SelectedRows> selected_rows_queue_;
  SimpleBlockQueue<framework::SelectedRows> selected_rows_return_queue_;
};

void RunServer(const SendRecvServerImpl &service,
               const std::string &server_address) {
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address << std::endl;
  server->Wait();
}

namespace paddle {
namespace operators {

class RecvOp : public framework::OperatorBase {
 public:
  RecvOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    constexpr RecvOpName = "RecvOp@SendRecvServerImpl";
    auto *var = scope.FindVar(RecvOpName);
    if (var == nullptr) {
      // create RPC server object if it is not inited.
      std::string endpoint = Attr<std::string>("endpoint");
      var = scope.Var(RecvOpName);
      SendRecvServerImpl *service = var->GetMutable<SendRecvServerImpl>();

      // start server in a thread in background
      std::thread server_thread(RunServer(*service, endpoit));
    }
    SendRecvServerImpl *service = var->Get<SendRecvServerImpl>();
    framework::LoDTensor &t = service->Get();
    // set graph input var
    auto *var = scope.Var(Input("X"));
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    // FIXME(typhoonzero): do not copy
    tensor->CopyFrom(t, dev_ctx.GetPlace(), dev_ctx);

    auto *block = Attr<framework::BlockDescBind *>("OptimizeBlock");
    auto *program = block->Program();
    framework::Executor executor(dev_ctx);
    // Run sub graph to get optimized tensor
    executor.Run(*program, &scope, block->ID(), false /*create_local_scope*/);

    auto *out_var = scope.FindVar("Out");
    service->Push(out_var->Get<framework::LoDTensor>());
  }
};

class RecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RecvOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be saved");
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv, ops::RecvOp, ops::RecvOpMaker);
