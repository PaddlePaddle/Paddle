// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
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

class DependOp : public framework::OperatorBase {
 public:
  DependOp(const std::string &type,
           const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    // NOTE(zhiqiu): depend op has empty compute, and it
    // can be skiped in the executor.
    OP_INOUT_CHECK(HasInputs("X"), "Input", "X", "Feed");
    OP_INOUT_CHECK(HasOutputs("Out"), "Output", "Out", "Feed");

    auto x_name = Input("X");
    auto out_name = Output("Out");
    PADDLE_ENFORCE_EQ(x_name,
                      out_name,
                      platform::errors::PreconditionNotMet(
                          "Input(X) and Output(Out) varibale should be the "
                          "same, but got Input is %s and Output is %s.",
                          x_name,
                          out_name));
    return;
  }
};

class DependOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

class DependOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Tensor, the dependence is added for.");
    AddInput("Dep", "The tensors that should be generated before X.")
        .AsDuplicable();
    AddOutput("Out", "Tensor, the same as input X");
    AddComment(R"DOC(
Depend Operator, allows to add explicit dependency between tensors.
For example, given two ops:
b = opA(a)
y = opB(x)

if tensor b and tensor x has some inner dependency, for example, x share data with b,
we need to add explicit dependency for x <- b, otherwise the these two operators may
be executed parellel in static graph. We can use depend op as below,

b = opA(a)
x = depend(x, b)
y = opB(x)

)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(DependNoNeedBufferVarsInferer, "X", "Dep");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    depend,
    ops::DependOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::DependOpProtoMaker,
    ops::DependOpShapeInference,
    ops::DependNoNeedBufferVarsInferer);
