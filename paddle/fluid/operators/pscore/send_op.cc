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

#include "paddle/fluid/distributed/fleet.h"
#include "paddle/fluid/distributed/service/communicator.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

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
    auto is_sparse = Attr<int>("is_sparse");
    auto table_id = Attr<int>("table_id");

    auto send_varnames = Attr<std::vector<std::string>>("send_varnames");

    // for common_dense_table, distributed_push_sparse op for push sparse in
    // async
    if (is_sparse == 0 && send_varnames.size() >= 1 &&
        send_varnames[0] != "@PS_STEP_COUNTER@") {
      auto fleet = paddle::distributed::FleetWrapper::GetInstance();
      std::vector<::std::future<int32_t>> status;
      fleet->PushDenseVarsAsync(scope, table_id, ins, &status, 0, -1);
    } else {
      auto* communicator = paddle::distributed::Communicator::GetInstance();
      if (communicator->Check(send_varnames)) {
        communicator->Send(ins, scope);
      }
    }
    // auto fleet = paddle::distributed::FleetWrapper::GetInstance();
    // if (is_sparse == 0) {
    //   std::vector<::std::future<int32_t>> status;
    //   fleet->PushDenseVarsAsync(scope, table_id, send_varnames, &status, 0,
    //   -1);
    // } else {
    //   std::vector<::std::future<int32_t>> status;
    //   fleet->PushSparseVarsAsync(scope, table_id, send_varnames[0], &status);
    // }
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
    AddAttr<int>("table_id", "table_id for send").SetDefault(0);
    AddAttr<int>("is_sparse",
                 "(int, default 0->Dense, 1->Sparse, 2->Distributed)")
        .SetDefault(0);
    AddAttr<std::vector<std::string>>(
        "send_varnames",
        "(vector<string>) "
        "the split output varnames to send to pserver")
        .SetDefault(std::vector<std::string>{});
  }
};

class SendOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    send, ops::SendOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::SendOpMaker, ops::SendOpShapeInference);
