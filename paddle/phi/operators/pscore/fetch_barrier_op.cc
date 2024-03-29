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

class FetchBarrierOp : public framework::OperatorBase {
 public:
  FetchBarrierOp(const std::string& type,
                 const framework::VariableNameMap& inputs,
                 const framework::VariableNameMap& outputs,
                 const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    VLOG(4) << "FetchBarrier Sync, do not need now";
  }
};

class FetchBarrierOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Any) Dummy inputs, used for control dependency")
        .AsDispensable()
        .AsDuplicable();
    AddOutput("Out", "(Any) Dummy outputs, used for control dependency")
        .AsDuplicable();
    AddComment(R"DOC(
SendBarrier operator

This operator will send a send barrier signal to list_and_serv op, so that
the Parameter Server would knew all variables have been sent.
)DOC");

    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.")
        .SetDefault({"127.0.0.1:6164"});
  }
};

class FetchBarrierOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fetch_barrier,
    ops::FetchBarrierOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::FetchBarrierOpMaker,
    ops::FetchBarrierOpShapeInference);
