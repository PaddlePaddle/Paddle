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
#include <iostream>
#include <ostream>
#include <thread>

#include <unistd.h>

#include "paddle/framework/data_type.h"
#include "paddle/framework/executor.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/detail/send_recv_impl.h"
#include "paddle/operators/detail/simple_block_queue.h"

namespace paddle {
namespace operators {

void RunServer(Server **rpc_server,
               std::shared_ptr<detail::SendRecvServerImpl> service,
               const std::string &server_address) {
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<Server> server(builder.BuildAndStart());
  *rpc_server = server.get();
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
      rpc_service_.reset(new detail::SendRecvServerImpl());
      std::string endpoint = Attr<std::string>("endpoint");
      server_thread_.reset(
          new std::thread(RunServer, &rpc_server_, rpc_service_, endpoint));
    }
  }

  virtual ~RecvOp() {
    rpc_server_->Shutdown();
    server_thread_->join();
  }

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    // FIXME(typhoonzero): no new scopes for every run.
    framework::Scope &recv_scope = scope.NewScope();
    auto param_list = Attr<std::vector<std::string>>("ParamList");
    auto grad_list = Attr<std::vector<std::string>>("GradList");
    size_t param_count = param_list.size();
    // TODO(typhoonzero): change this to a while_op for every cluster-batch.
    while (true) {
      // TODO(typhoonzero): get from multiple trainers.
      for (size_t i = 0; i < param_count; ++i) {
        // blocking get one var from client.
        const detail::TensorWithName &v = rpc_service_->Get();
        auto grad_var_name = v.first;
        auto it = std::find(grad_list.begin(), grad_list.end(), grad_var_name);
        std::string param_var_name;
        if (it != grad_list.end()) {
          param_var_name = param_list[it - grad_list.begin()];
        }
        VLOG(10) << "recved grad: " << grad_var_name
                 << " updating param: " << param_var_name;
        auto *var = recv_scope.Var(grad_var_name);
        auto *tensor = var->GetMutable<framework::LoDTensor>();
        // FIXME(typhoonzero): do not copy
        framework::CopyFrom(v.second, dev_ctx.GetPlace(), dev_ctx, tensor);
      }

      std::string program_str = Attr<std::string>("OptimizeProgram");
      framework::ProgramDesc program_desc;
      program_desc.ParseFromString(program_str);
      framework::ProgramDescBind program(program_desc);
      framework::Executor executor(dev_ctx);
      // Run sub graph to get optimized tensor
      try {
        executor.Run(program, &recv_scope, 0, /*global_block*/
                     false /*create_local_scope*/, false /*create_vars*/);
      } catch (std::exception &e) {
        LOG(ERROR) << "run sub program error " << e.what();
      }

      for (size_t i = 0; i < param_count; ++i) {
        auto *out_var = recv_scope.FindVar(param_list[i]);
        detail::TensorWithName out;
        out.first = param_list[i];
        out.second = out_var->Get<framework::LoDTensor>();
        rpc_service_->Push(out);
      }
    }  // while(true)
  }

 protected:
  // grpc server instance to track status and gracefully shutdown.
  // borrow an pointer from server thread.
  Server *rpc_server_{nullptr};
  // grpc send/recv service implement to register.
  std::shared_ptr<detail::SendRecvServerImpl> rpc_service_;
  std::shared_ptr<std::thread> server_thread_;
};

class RecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RecvOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("RX", "(Tensor) Input tensor to be optimized").AsDuplicable();
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<std::string>("OptimizeProgram", "type string",
                         "Serialized ProgramDesc string for recv to run.");
    AddAttr<std::vector<std::string>>(
        "ParamList", "type list of string",
        "grad->param name mapping to find which param to optimize.");
    AddAttr<std::vector<std::string>>(
        "GradList", "type list of string",
        "grad->param name mapping to find which param to optimize.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv, ops::RecvOp, ops::RecvOpMaker);
