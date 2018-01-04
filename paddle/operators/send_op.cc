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

//#include "paddle/operators/detail/grpc_client.h"
//#include "paddle/operators/detail/send_recv_impl.h"
//#include "paddle/operators/detail/simple_block_queue.h"
#include <future>
#include "paddle/operators/detail/grpc_client2.h"

namespace paddle {
namespace operators {

bool Send(const std::vector<std::string>& ep, const framework::Scope* scope,
          const std::vector<std::string>& ins, int64_t time_out) {
  detail::RPCClients c;
  for (size_t i = 0; i < ins.size(); i++) {
    c.AsyncSendVariable(ep[i], scope, ins[i], time_out);
  }

  bool ok = true;
  while (true) {
    detail::SendStatus s;
    if (!c.Proceed(s)) {
      LOG(ERROR) << "Send meets CompletionQueue error";
      return false;
    }

    // TODO(gongwb): add more retry?
    if (s.error != "") {
      ok = false;
      LOG(ERROR) << "sync update variable error:" << s.String();
    } else {
      VLOG(3) << "sync update variable ok:" << s.String();
    }
  }

  return ok;
}

bool Get(const std::vector<std::string>& ep, const framework::Scope* scope,
         const std::vector<std::string>& ins, int64_t time_out) {
  detail::RPCClients c;
  for (size_t i = 0; i < ins.size(); i++) {
    c.AsyncGetVariable(ep[i], scope, ins[i], time_out);
  }

  bool ok = true;
  while (true) {
    detail::SendStatus s;
    if (!c.Proceed(s)) {
      LOG(ERROR) << "Get meets CompletionQueue error";
      return false;
    }

    // TODO(gongwb): add more retry?
    if (s.error != "") {
      ok = false;
      LOG(ERROR) << "sync update variable error:" << s.String();
    } else {
      VLOG(3) << "sync update variable ok:" << s.String();
    }
  }

  return ok;
}

// TODO(gongwb): add more attrs to support more send pattern.
class SendOp : public framework::OperatorBase {
 public:
  SendOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  bool SyncUpdate(const std::vector<std::string>& eps,
                  const framework::Scope* scope,
                  const std::vector<std::string>& ins,
                  const std::vector<std::string>& outs) const {
    int64_t send_timeout = 5000 * 1000;
    int64_t get_timeout = 1800 * 1000;
    auto send_thread = std::async(Send, eps, scope, ins, send_timeout);
    auto get_thread = std::async(Get, eps, scope, outs, get_timeout);

    auto send_ok = send_thread.get();
    auto get_ok = get_thread.get();

    return (send_ok && get_ok);
  }

  void Run(const framework::Scope& scope,
           const platform::Place& dev_place) const override {
    auto ins = Inputs("X");
    auto outs = Outputs("Out");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    SyncUpdate(epmap, &scope, ins, outs);
  }
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be send").AsDuplicable();
    AddOutput("Out", "(Tensor) Output tensor to get from server")
        .AsDuplicable();
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
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
