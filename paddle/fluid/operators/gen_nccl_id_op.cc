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

#include <nccl.h>
#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/detail/grpc_server.h"

namespace paddle {
namespace operators {

class GenNCCLIdOp : public framework::OperatorBase {
 public:
  GenNCCLIdOp(const std::string& type, const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // put nccl id in CPUPlace
    auto& dev_ctx = *pool.Get(platform::CPUPlace());
    int trainer_id = Attr<int>("trainer_id");
    framework::Scope& local_scope = scope.NewScope();

    if (trainer_id == 0) {
      GenerateAndSend(&local_scope, dev_ctx);
    } else {
      GetIdByServer(&local_scope, dev_ctx);
    }
  }

 private:
  void GenerateAndSend(framework::Scope* scope,
                       const platform::DeviceContext& dev_ctx) const {
    auto var = scope->FindVar("NCCLID");
    PADDLE_ENFORCE_NOT_NULL(var);
    auto id = var->GetMutable<ncclUniqueId>();
    platform::dynload::ncclGetUniqueId(id);

    std::vector<std::string> endpoint_list =
        Attr<std::vector<std::string>>("endpoint_list");
    detail::RPCClient client;
    for (auto& ep : endpoint_list) {
      VLOG(3) << "sending nccl id to " << ep;
      client.AsyncSendVariable(ep, dev_ctx, *scope, "NCCLID");
    }
    client.Wait();
    VLOG(3) << "sending completed...";
  }

  void GetIdByServer(framework::Scope* scope,
                     const platform::DeviceContext& dev_ctx) const {
    std::string endpoint = Attr<std::string>("endpoint");
    rpc_service_.reset(new detail::AsyncGRPCServer(endpoint, true));
    framework::ProgramDesc empty_program;
    framework::Executor executor(dev_ctx.GetPlace());
    rpc_service_->SetScope(scope);
    rpc_service_->SetDevCtx(&dev_ctx);
    rpc_service_->SetProgram(&empty_program);
    rpc_service_->SetExecutor(&executor);

    server_thread_.reset(new std::thread(std::bind(
        &detail::AsyncGRPCServer::RunSyncUpdate, rpc_service_.get())));
    rpc_service_->SetCond(0);
    VLOG(3) << "start getting nccl id from trainer 0...";
    auto recv = rpc_service_->Get();
    VLOG(3) << "got nccl id and stop server...";
    // rpc_service_->SetCond(1);
    // rpc_service_->ShutDown();
    rpc_service->Push(LISTEN_TERMINATE_MESSAGE);
    VLOG(3) << "rpc server stopped";
    // TODO(wuyi): reinit nccl communicators
  }

 protected:
  mutable std::shared_ptr<detail::AsyncGRPCServer> rpc_service_;
  mutable std::shared_ptr<std::thread> server_thread_;
};

class GenNCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  GenNCCLIdOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("NCCLID", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
GenNCCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string), e.g. 127.0.0.1:6175 "
                         "current listen endpoint");
    AddAttr<std::vector<std::string>>(
        "endpoint_list",
        "['trainer1_ip:port', 'trainer2_ip:port', ...] "
        "list of trainer endpoints start from trainer 1")
        .SetDefault({});
    AddAttr<int>("trainer_id",
                 "(int default 0) "
                 "The index of the trainer in distributed training.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gen_nccl_id, ops::GenNCCLIdOp, ops::GenNCCLIdOpMaker);
