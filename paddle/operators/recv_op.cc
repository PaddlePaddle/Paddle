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
#include "paddle/framework/executer.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/detail/send_recv_impl.h"
#include "paddle/operators/detail/simple_block_queue.h"

namespace paddle {
namespace operators {

void RunServer(std::shared_ptr<detail::SendRecvServerImpl> service,
               const std::string &server_address) {
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address << std::endl;
  server->Wait();
}

class RecvOp : public framework::OperatorBase {
 public:
  RecvOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    if (!rpc_service_) {
      std::string endpoint = Attr<std::string>("endpoint");
      std::thread server_thread(RunServer(rpc_service_, endpoint));
    }
  }
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    // blocking get one var from client.
    const framework::LoDTensor &t = rpc_service_->Get();
    // set graph input var
    auto *var = scope.FindVar(Input("X"));
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    // FIXME(typhoonzero): do not copy
    tensor->CopyFrom(t, dev_ctx.GetPlace(), dev_ctx);

    auto *block = Attr<framework::BlockDescBind *>("OptimizeBlock");
    auto *program = block->Program();
    framework::Executor executor(dev_ctx);
    // Run sub graph to get optimized tensor
    executor.Run(*program, &scope, block->ID(), false /*create_local_scope*/);

    auto *out_var = scope.FindVar("Out");
    // push back
    rpc_service_->Push(out_var->Get<framework::LoDTensor>());
  }

 protected:
  std::shared_ptr<detail::SendRecvServerImpl> rpc_service_;
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
