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

#ifdef PADDLE_WITH_GRPC
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/detail/grpc_server.h"
#else
#include "paddle/fluid/operators/detail/brpc_client.h"
#include "paddle/fluid/operators/detail/brpc_server.h"
#endif

#include "paddle/fluid/operators/detail/request_handler_impl.h"
#include "paddle/fluid/platform/nccl_helper.h"

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
    auto var = scope->FindVar(NCCL_ID_VARNAME);
    PADDLE_ENFORCE_NOT_NULL(var);
    auto id = var->GetMutable<ncclUniqueId>();
    PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(id));

    std::vector<std::string> endpoint_list =
        Attr<std::vector<std::string>>("endpoint_list");
#ifdef PADDLE_WITH_GRPC
    detail::RPCClient* client =
        detail::RPCClient::GetInstance<detail::GRPCClient>();
#else
    detail::RPCClient* client =
        detail::RPCClient::GetInstance<detail::BRPCClient>();
#endif

    for (auto& ep : endpoint_list) {
      VLOG(3) << "sending nccl id to " << ep;
      client->AsyncSendVar(ep, dev_ctx, *scope, NCCL_ID_VARNAME);
    }
    client->Wait();
    VLOG(3) << "sending completed...";
  }

  void GetIdByServer(framework::Scope* scope,
                     const platform::DeviceContext& dev_ctx) const {
    std::string endpoint = Attr<std::string>("endpoint");
    // NOTE: Can not use unique_ptr here because the default
    // deleter will call GRPC Server's base class's dtor and
    // that will cause a wired crash.
    detail::RequestSendHandler rpc_h(true);
    std::unique_ptr<detail::RPCServer> rpc_service(nullptr);
#ifdef PADDLE_WITH_GRPC
    rpc_service.reset(new detail::AsyncGRPCServer(endpoint, 1));
#else
    rpc_service.reset(new detail::AsyncBRPCServer(endpoint, 1));
#endif

    rpc_service->RegisterRPC(detail::kRequestSend, &rpc_h);
    rpc_h.SetRPCServer(rpc_service.get());

    framework::ProgramDesc empty_program;
    framework::Executor executor(dev_ctx.GetPlace());
    rpc_h.SetScope(scope);
    rpc_h.SetDevCtx(&dev_ctx);
    rpc_h.SetProgram(&empty_program);
    rpc_h.SetExecutor(&executor);

    std::thread server_thread(
        std::bind(&detail::RPCServer::StartServer, rpc_service.get()));

    rpc_service->SetCond(detail::kRequestSend);
    VLOG(3) << "start getting nccl id from trainer 0...";
    rpc_service->WaitBarrier(detail::kRequestSend);
    VLOG(3) << "got nccl id and stop server...";
    rpc_service->ShutDown();
    VLOG(3) << "rpc server stopped";
    server_thread.join();
  }
};

class GenNCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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
