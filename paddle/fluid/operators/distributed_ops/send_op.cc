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

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/communicator.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"
#include "paddle/fluid/operators/distributed/rpc_common.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
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

    auto epmap = Attr<std::vector<std::string>>("epmap");
    int sync_send = Attr<int>("sync_mode");
    auto trainer_id = Attr<int>("trainer_id");

    auto send_varnames = Attr<std::vector<std::string>>("send_varnames");
    auto height_sections = Attr<std::vector<int64_t>>("sections");

    if (send_varnames.size() > 0) {
      PADDLE_ENFORCE_EQ(ins.size(), 1, "");
      if (distributed::Communicator::GetInstance() == nullptr) {
        auto send_functor = distributed::ParameterSend<float>();
        auto rpc_ctx = distributed::RpcContext(ins[0], send_varnames, epmap,
                                               height_sections, trainer_id);
        send_functor(rpc_ctx, scope, true);
      } else {
        distributed::Communicator::GetInstance()->Send(ins[0], scope);
      }
    } else {
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto& ctx = *pool.Get(place);

      distributed::RPCClient* rpc_client =
          distributed::RPCClient::GetInstance<RPCCLIENT_T>(trainer_id);

      std::vector<distributed::VarHandlePtr> rets;
      for (size_t i = 0; i < ins.size(); i++) {
        if (NeedSend(scope, ins[i])) {
          VLOG(3) << "sending " << ins[i] << " to " << epmap[i];
          rets.push_back(
              rpc_client->AsyncSendVar(epmap[i], ctx, scope, ins[i]));
        } else {
          VLOG(3) << "don't send no-initialied variable: " << ins[i];
        }
      }
      if (sync_send) {
        for (size_t i = 0; i < rets.size(); i++) {
          VLOG(7) << "before sync_send " << ins[i] << "from " << epmap[i];
          PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
          VLOG(7) << "after sync_send " << ins[i] << "from " << epmap[i];
        }
      }
    }
  }
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor, SelectedRows) Input variables to be sent")
        .AsDuplicable();
    AddOutput("Out", "(Any) Dummy outputs, used for control dependency")
        .AsDuplicable();
    AddComment(R"DOC(
Send operator

This operator will send variables to listen_and_serve op at the parameter server.
)DOC");
    AddAttr<int>("sync_mode",
                 "(int, default 0)"
                 "sync send or async send.")
        .SetDefault(0);
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<std::vector<int64_t>>("sections",
                                  "(vector<int>) "
                                  "the length of each output along the "
                                  "specified axis.")
        .SetDefault(std::vector<int64_t>{});
    AddAttr<std::vector<std::string>>(
        "send_varnames",
        "(vector<string>) "
        "the splited output varnames to send to pserver")
        .SetDefault(std::vector<std::string>{});
    AddAttr<int>("num",
                 "(int, default 0)"
                 "Number of sub-tensors. This must evenly divide "
                 "Input.dims()[axis]")
        .SetDefault(0);
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
