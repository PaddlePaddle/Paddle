/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/communicator.h"
#include "paddle/fluid/operators/distributed/communicator_common.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class SendAndRecvOp : public framework::OperatorBase {
 public:
  SendAndRecvOp(const std::string& type,
                const framework::VariableNameMap& inputs,
                const framework::VariableNameMap& outputs,
                const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    auto send_var_name = Attr<std::string>("send_var_name");
    auto recv_var_name = Attr<std::string>("recv_var_name");
    auto epmap = Attr<std::string>("endpoint");
    auto trainer_id = Attr<int>("trainer_id");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(trainer_id);

    distributed::VarHandlePtr rets = rpc_client->AsyncSendAndRecv(
        epmap, ctx, scope, send_var_name, recv_var_name);
    rets->Wait();
  }
};

class SendAndRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "Tensor Input variable to be sent").AsDuplicable();
    AddOutput("Out", "Tensor Output varibale to be recv").AsDuplicable();
    AddAttr<std::string>("send_var_name", "Send Tensor's name")
        .SetDefault(std::string(""));
    AddAttr<std::string>("recv_var_name", "Recv Tensor's name")
        .SetDefault(std::string(""));
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::string>("endpoint", "Server endpoint")
        .SetDefault({"127.0.0.1:6164"});
    AddComment(R"DOC(
    SendAndRecv operator

    This operator will send variables to listen_and_serve op at the parameter server.
    And recv variable from parameter server of send variable's scope.
    )DOC");
  }
};

class SendAndRecvOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    send_and_recv, ops::SendAndRecvOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::SendAndRecvOpMaker, ops::SendAndRecvOpShapeInference);