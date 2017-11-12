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

#include <stdint.h>
#include <sys/stat.h>
#include <ostream>
#include <thread>

#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
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
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    constexpr SendOpName = "SendOp@RPCClient";
    auto *var = scope.FindVar(SendOpName);
    if (var == nullptr) {
      // create RPC server object if it is not inited.
      std::string endpoint = Attr<std::string>("endpoint");
      var = scope.Var(SendOpName);
      RPCClient *client = var->GetMutable<RPCClient>();
    }
    RPCClient *client = var->Get<RPCClient>();

    auto iname = Input("X");
    auto oname = Output("Out");
    auto *var = scope.FindVar(iname);
    auto *tensor = var->Get<framework::LoDTensor>();
    // call sync send
    auto *optimized_tensor = client->SendTensor(*tensor);
    // FIXME(typhoonzero): do not copy
    auto *out_var = scope.FindVar(oname);
    out_var->GetMutable<framework::LoDTensor>();
    out_var->CopyFrom(*optimized_tensor, dev_ctx.GetPlace(), dev_ctx);
  }
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be saved");
    AddOutput("Out", "(Tensor) Output fetched from server");
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send, ops::SendOp, ops::SendOpMaker);
