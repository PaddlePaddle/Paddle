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

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);
    for (size_t i = 0; i < ins.size(); i++) {
      VLOG(3) << "sending " << ins[i] << " to " << epmap[i];
      client_.AsyncSendVariable(epmap[i], ctx, scope, ins[i]);
    }
    PADDLE_ENFORCE(client_.Wait());

    std::set<std::string> epset(epmap.begin(), epmap.end());
    for (auto& ep : epset) {
      VLOG(3) << "batch barrier, ep: " << ep;
      client_.AsyncBatchBarrier(ep);
    }
    PADDLE_ENFORCE(client_.Wait());

    for (size_t i = 0; i < outs.size(); i++) {
      VLOG(3) << "getting " << outs[i] << " from " << epmap[i];
      client_.AsyncGetVariable(epmap[i], ctx, scope, outs[i]);
    }

    PADDLE_ENFORCE(client_.Wait());
  }

 private:
  mutable detail::RPCClient client_;
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be sent").AsDuplicable();
    AddOutput("Out", "(Tensor) Output tensor to be received from server")
        .AsDuplicable();
    AddComment(R"DOC(
Send operator

This operator will send tensor to recv_op at the parameter server.
)DOC");
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
