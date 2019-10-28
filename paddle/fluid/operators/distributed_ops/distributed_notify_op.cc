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

class DistributedNotifyOp : public framework::OperatorBase {
 public:
  DistributedNotifyOp(const std::string& type,
                      const framework::VariableNameMap& inputs,
                      const framework::VariableNameMap& outputs,
                      const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    auto ins = Inputs("X");

    auto epmap = Attr<std::vector<std::string>>("epmap");
    auto trainer_id = Attr<int>("trainer_id");

    auto send_varnames = Attr<std::vector<std::string>>("send_varnames");
    auto height_sections = Attr<std::vector<int64_t>>("sections");

    if (send_varnames.size() > 0) {
      if (ins.size() > 1) {
        distributed::Communicator::GetInstance()->Send(ins, send_varnames,
                                                       scope);
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
          VLOG(3) << "notifying " << ins[i] << " to " << epmap[i];
          rets.push_back(
              rpc_client->AsyncDistributeNotify(epmap[i], ctx, scope, ins[i]));
        } else {
          VLOG(3) << "don't notify no-initialied variable: " << ins[i];
        }
      }
      for (size_t i = 0; i < rets.size(); i++) {
        VLOG(7) << "before sync_notify " << ins[i] << "from " << epmap[i];
        PADDLE_ENFORCE_NE(rets[i]->Wait(), 0U, "internal error in RPCClient");
        VLOG(7) << "after sync_notify " << ins[i] << "from " << epmap[i];
      }
    }
  }
};

class DistributedNotifyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor, SelectedRows) Input variables to be sent")
        .AsDuplicable();
    AddInput("Y", "(Any) Dummy inputs, used for control dependency")
        .AsDuplicable();
    AddOutput("Out", "(Any) Dummy outputs, used for control dependency")
        .AsDuplicable();
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
    AddAttr<bool>("merge_add",
                  "(bool, default 0)"
                  "merge method, true represent add, false represent average")
        .SetDefault(false);
    AddComment(R"DOC(
DistributeNotify operator

This operator will send variables to listen_and_serve op at the parameter server. it's basic function is same as send op.
)DOC");
  }
};

class DistributedNotifyOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(distributed_notify, ops::DistributedNotifyOp,
                  paddle::framework::EmptyGradOpMaker,
                  ops::DistributedNotifyOpMaker,
                  ops::DistributedNotifyOpShapeInference);
