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

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/send_recv_util.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class SendOp : public framework::OperatorBase {
 public:
  SendOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    auto ins = Inputs("X");
    auto outs = Outputs("Out");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    std::vector<std::string> endpoints =
        Attr<std::vector<std::string>>("endpoints");

    bool sync_mode = Attr<bool>("sync_mode");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);

    // For profiling
    platform::RecordEvent record_event(Type(), &ctx);

    auto rpc_client = detail::RPCClient::GetInstance();

    for (size_t i = 0; i < ins.size(); i++) {
      if (NeedSend(scope, ins[i])) {
        VLOG(3) << "sending " << ins[i] << " to " << epmap[i];
        rpc_client->AsyncSendVariable(epmap[i], ctx, scope, ins[i]);
      } else {
        VLOG(3) << "don't send no-initialied variable: " << ins[i];
      }
    }
    PADDLE_ENFORCE(rpc_client->Wait());

    if (sync_mode) {
      for (auto& ep : endpoints) {
        VLOG(3) << "batch barrier, ep: " << ep;
        rpc_client->AsyncSendBatchBarrier(ep);
      }
      PADDLE_ENFORCE(rpc_client->Wait());
    }

    if (outs.size() > 0) {
      for (size_t i = 0; i < outs.size(); i++) {
        VLOG(2) << "getting " << outs[i] << " from " << epmap[i];
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
  }
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Input tensor to be sent").AsDuplicable();
    AddOutput("Out", "(Tensor) Output tensor to be received from server")
        .AsDuplicable();
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
    AddAttr<bool>("sync_mode", "work in sync_mode or not").SetDefault(true);
  }
};

class SendOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send, ops::SendOp, paddle::framework::EmptyGradOpMaker,
                  ops::SendOpMaker, ops::SendOpShapeInference);
