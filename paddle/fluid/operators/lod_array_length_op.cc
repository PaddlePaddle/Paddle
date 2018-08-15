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

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class LoDArrayLengthOp : public framework::OperatorBase {
 public:
  LoDArrayLengthOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensorArray>();
    auto &out =
        *scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();
    out.Resize({1});
    auto cpu = platform::CPUPlace();
    *out.mutable_data<int64_t>(cpu) = static_cast<int64_t>(x.size());
  }
};

class LoDArrayLengthProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensorArray) The input tensor array.");
    AddOutput("Out", "(Tensor) 1x1 CPU Tensor of length, int64_t");
    AddComment(R"DOC(
LoDArrayLength Operator.

This operator obtains the length of lod tensor array:

$$Out = len(X)$$

NOTE: The output is a CPU Tensor since the control variable should be only in
CPU and the length of LoDTensorArray should be used as control variables.

)DOC");
  }
};

class LoDArrayLengthInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"));
    PADDLE_ENFORCE(context->HasOutput("Out"));
    context->SetOutputDim("Out", {1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lod_array_length, ops::LoDArrayLengthOp,
                  ops::LoDArrayLengthInferShape, ops::LoDArrayLengthProtoMaker,
                  paddle::framework::EmptyGradOpMaker);
