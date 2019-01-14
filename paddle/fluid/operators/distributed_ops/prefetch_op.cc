/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"

namespace paddle {
namespace operators {

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

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(
            Attr<int>("trainer_id"));

    std::vector<distributed::VarHandlePtr> rets;
    for (size_t i = 0; i < ins.size(); i++) {
      if (NeedSend(scope, ins[i])) {
        VLOG(3) << "sending " << ins[i] << " to " << epmap[i] << " to get "
                << outs[i] << " back";
        rets.push_back(rpc_client->AsyncPrefetchVar(epmap[i], ctx, scope,
                                                    ins[i], outs[i]));
      } else {
        VLOG(3) << "don't send no-initialied variable: " << ins[i];
      }
    }
    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
    }
  }
};

class PrefetchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(LoDTensor) Input Id variables to be sent").AsDuplicable();
    AddOutput("Out",
              "(LoDTensor) result "
              "to be fetched from parameter server")
        .AsDuplicable();
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<std::string>>(
        "epmap",
        "(string vector, default 127.0.0.1:6164)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({"127.0.0.1:6164"});
    AddComment(R"DOC(
Prefetch operator

This operator will send Ids variables to listen_and_serve op at
the parameter server and fetch result back.
)DOC");
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
                  ops::PrefetchOpShapeInference);
