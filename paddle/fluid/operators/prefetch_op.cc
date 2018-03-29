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

namespace paddle {
namespace operators {
static bool NeedSend(const framework::Scope& scope,
                     const std::string& varname) {
  auto* var = scope.FindVar(varname);
  PADDLE_ENFORCE_NOT_NULL(var, "Can not find variable '%s' in the send side.",
                          varname);
  if (var->IsType<framework::LoDTensor>()) {
    return var->Get<framework::LoDTensor>().numel() > 0;
  } else if (var->IsType<framework::SelectedRows>()) {
    return var->Get<framework::SelectedRows>().value().numel() > 0;
  } else {
    PADDLE_THROW(
        "Variable type in send side should be in "
        "[LodTensor, SelectedRows]");
  }
  return false;
}

class PrefetchOp : public framework::OperatorBase {
 public:
  PrefetchOp(const std::string& type, const framework::VariableNameMap& inputs,
             const framework::VariableNameMap& outputs,
             const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    auto ins = Inputs("X");
    auto outs = Outputs("Out");

    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);

    auto client_var_name = Output("RPCClient");
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(client_var_name),
                            "Can not find variable '%s' in the scope.",
                            client_var_name);
    auto* client_var = scope.FindVar(client_var_name);
    detail::RPCClient* rpc_client = client_var->GetMutable<detail::RPCClient>();

    for (size_t i = 0; i < ins.size(); i++) {
      if (NeedSend(scope, ins[i])) {
        VLOG(3) << "sending " << ins[i] << " to " << epmap[i] << "to get "
                << outs[i] << "back";
        rpc_client->AsyncPrefetchVariable(epmap[i], ctx, scope, ins[i],
                                          outs[i]);
      } else {
        VLOG(3) << "don't send no-initialied variable: " << ins[i];
      }
    }
    rpc_client->Wait();
  }
};

class PrefetchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PrefetchOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(LoDTensor) Input Id variables to be sent").AsDuplicable();
    AddOutput("RPCClient",
              "(RPCClient) The RPC client object which will be"
              "initialized at most once.");
    AddOutput("Out",
              "(SelectedRows) result "
              "to be fetched from parameter server")
        .AsDuplicable();
    AddComment(R"DOC(
Send operator

This operator will send Ids variables to listen_and_serve op at the parameter server and fetch result back.
)DOC");
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({"127.0.0.1:6164"});
  }
};

class PrefetchOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    auto out_var_name = op_desc.Output("RPCClient").front();
    auto& out_var = block->FindRecursiveOrCreateVar(out_var_name);
    auto var_type = framework::proto::VarType::RAW;
    out_var.SetType(var_type);
  }
};

class PrefetchOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(prefetch, ops::PrefetchOp,
                  paddle::framework::EmptyGradOpMaker, ops::PrefetchOpMaker,
                  ops::PrefetchOpVarTypeInference,
                  ops::PrefetchOpShapeInference);
