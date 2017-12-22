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

#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"

#include "paddle/operators/detail/send_recv_impl.h"
#include "paddle/operators/detail/simple_block_queue.h"

namespace paddle {
namespace operators {

// TODO(typhoonzero): this is a simple implementation which only send
// one tensor
class SendOp : public framework::OperatorBase {
 public:
  SendOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    // init client when the operator is created at runtime.
    std::vector<std::string> endpoints =
        Attr<std::vector<std::string>>("endpoints");
    for (auto ep : endpoints) {
      client_map_[ep].reset(new detail::RPCClient(
          grpc::CreateChannel(ep, grpc::InsecureChannelCredentials())));
    }
  }
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto ins = Inputs("X");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    // TODO(typhoonzero): use async calls to send multiple variable asyncly.
    for (size_t i = 0; i < ins.size(); ++i) {
      bool ret = client_map_[epmap[i]]->SendVariable(scope, ins[i]);
      if (!ret) {
        LOG(ERROR) << "send variable error: " << ins[i];
      }
    }
    // TODO(typhoonzero): support async optimization
    client_map_[epmap[0]]->Wait();
    for (size_t i = 0; i < ins.size(); ++i) {
      bool ret = client_map_[epmap[i]]->GetVariable(scope, ins[i]);
      if (!ret) {
        LOG(ERROR) << "GetVariable error: " << ins[i];
      }
    }
  }

 protected:
  mutable std::unordered_map<std::string, std::shared_ptr<detail::RPCClient>>
      client_map_;
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be send").AsDuplicable();
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
)DOC");
    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.");
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send, ops::SendOp, ops::SendOpMaker);
