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
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/distributed/collective_client.h.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
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

    distributed::CollectiveClient::BroadCast(endpoint_list, dev_ctx, scope,
                                             NCCL_ID_VARNAME);
  }

  void GetIdByServer(framework::Scope* scope,
                     const platform::DeviceContext& dev_ctx) const {
    std::string endpoint = Attr<std::string>("endpoint");
    std::vector<std::string> endpoint_list =
        Attr<std::vector<std::string>>("endpoint_list");
    operators::distributed::CollectiveServer* server =
        operators::distributed::CollectiveServer::GetInstance(
            end_point, endpoint_list.size());

    server->RegisterSendRPC(scope, &dev_ctx);
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
        "list of trainer endpoints start from trainer 1");
    AddAttr<int>("trainer_id",
                 "(int default 0) "
                 "The index of the trainer in distributed training.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gen_nccl_id, ops::GenNCCLIdOp, ops::GenNCCLIdOpMaker);
