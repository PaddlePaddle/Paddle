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

#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#include <future>
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class RecvOp : public framework::OperatorBase {
 public:
  RecvOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    auto outs = Outputs("Out");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    std::vector<std::string> endpoints =
        Attr<std::vector<std::string>>("endpoints");
    auto client_var_name = Output("RPCClient");
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(client_var_name),
                            "Can not find variable '%s' in the scope.",
                            client_var_name);

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);

    platform::RecordEvent record_event(Type(), &ctx);
    auto* client_var = scope.FindVar(client_var_name);
    detail::RPCClient* rpc_client = client_var->GetMutable<detail::RPCClient>();

    for (size_t i = 0; i < outs.size(); i++) {
      VLOG(3) << "getting " << outs[i];
      rpc_client->AsyncGetVariable(epmap[i], ctx, scope, outs[i]);
    }
    PADDLE_ENFORCE(rpc_client->Wait());

    // tell pservers that current trainer have called fetch
    for (auto& ep : endpoints) {
      VLOG(2) << "send fetch barrier, ep: " << ep;
      rpc_client->AsyncSendFetchBarrier(ep);
    }
    PADDLE_ENFORCE(rpc_client->Wait());
  }
};

class RecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RecvOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "(Tensor) Variables to get from server.").AsDuplicable();
    AddOutput("RPCClient",
              "(RPCClient) The RPC client object which will be"
              "initialized at most once.");
    AddComment(R"DOC(
Recv operator

This operator can get variables from server side.
)DOC");
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints list")
        .SetDefault({"127.0.0.1:6164"});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv, ops::RecvOp, ops::RecvOpMaker);
