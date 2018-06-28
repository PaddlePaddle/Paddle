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
#include "paddle/fluid/operators/detail/macros.h"

#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class SendBarrierOp : public framework::OperatorBase {
 public:
  SendBarrierOp(const std::string& type,
                const framework::VariableNameMap& inputs,
                const framework::VariableNameMap& outputs,
                const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    std::vector<std::string> eps = Attr<std::vector<std::string>>("endpoints");
    bool sync_mode = Attr<bool>("sync_mode");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);
    // For profiling
    platform::RecordEvent record_event(Type(), &ctx);

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>();

    VLOG(3) << "SendBarrierOp sync_mode:" << sync_mode;

    // need to wait before sending send_barrier message
    rpc_client->Wait();
    if (sync_mode) {
      for (auto& ep : eps) {
        VLOG(3) << "send barrier, ep: " << ep;
        rpc_client->AsyncSendBatchBarrier(ep);
      }
      rpc_client->Wait();
    }
  }
};

class SendBarrierOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddComment(R"DOC(
SendBarrier operator

This operator will send a send barrier signal to list_and_serv op, so that
the Parameter Server would knew all variables have been sent.
)DOC");

    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<bool>("sync_mode", "work in sync_mode or not").SetDefault(true);
  }
};

class SendBarrierOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send_barrier, ops::SendBarrierOp,
                  paddle::framework::EmptyGradOpMaker, ops::SendBarrierOpMaker,
                  ops::SendBarrierOpShapeInference);
