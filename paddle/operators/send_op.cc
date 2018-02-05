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

#include <ostream>

#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"

#include <future>
#include "paddle/operators/detail/grpc_client.h"

namespace paddle {
namespace operators {

class SendOp : public framework::OperatorBase {
 public:
  SendOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope& scope,
           const platform::Place& place) const override {
    auto ins = Inputs("X");
    auto outs = Outputs("Out");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    std::vector<std::string> endpoints =
        Attr<std::vector<std::string>>("endpoints");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);

    auto client_var_name = Output("RPCClient");
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(client_var_name),
                            "Can not find variable '%s' in the scope.",
                            client_var_name);
    auto* client_var = scope.FindVar(client_var_name);
    detail::RPCClient* rpc_client = client_var->GetMutable<detail::RPCClient>();

    for (size_t i = 0; i < ins.size(); i++) {
      VLOG(3) << "sending " << ins[i] << " to " << epmap[i];
      rpc_client->AsyncSendVariable(epmap[i], ctx, scope, ins[i]);
    }
    PADDLE_ENFORCE(rpc_client->Wait());

    for (auto& ep : endpoints) {
      VLOG(3) << "batch barrier, ep: " << ep;
      rpc_client->AsyncSendBatchBarrier(ep);
    }
    PADDLE_ENFORCE(rpc_client->Wait());

    if (outs.size() > 0) {
      for (size_t i = 0; i < outs.size(); i++) {
        VLOG(3) << "getting " << outs[i] << " from " << epmap[i];
        rpc_client->AsyncGetVariable(epmap[i], ctx, scope, outs[i]);
      }
      PADDLE_ENFORCE(rpc_client->Wait());
    }
  }
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be sent").AsDuplicable();
    AddOutput("Out", "(Tensor) Output tensor to be received from server")
        .AsDuplicable();
    AddOutput("RPCClient",
              "(RPCClient) The RPC client object which is"
              "initialized at most once.");
    AddComment(R"DOC(
Send operator

This operator will send tensor to recv_op at the parameter server.
)DOC");
    // TODO(typhoonzero): remove this attr generate de-duplicated vector from
    // epmap when initializing.
    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send, ops::SendOp, ops::SendOpMaker);
