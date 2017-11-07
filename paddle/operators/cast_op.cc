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

#include "paddle/operators/cast_op.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class CastOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CastOpProtoMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of cast op");
    AddOutput("Out", "The output tensor of cast op");
    AddAttr<int>("out_data_type", "output data type");
    AddAttr<int>("in_data_type", "input data type");
    AddComment(R"DOC(
Cast Operator.

This Operator casts the input tensor to another data type and
returns tha Output Tensor.

)DOC");
  }
};

class CastOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"), "The input of cast op must be set");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "The output of cast op must be set");
    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }
};

class CastOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto grad = new framework::OpDescBind();
    grad->SetType("cast");
    grad->SetInput("X", OutputGrad("Out"));
    grad->SetOutput("Out", InputGrad("X"));
    grad->SetAttr("out_data_type", GetAttr("in_data_type"));
    grad->SetAttr("in_data_type", GetAttr("out_data_type"));
    return std::unique_ptr<framework::OpDescBind>(grad);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUPlace;
REGISTER_OP_WITH_KERNEL(cast, ops::CastOpGradMaker, ops::CastOpInferShape,
                        ops::CastOpProtoMaker);
REGISTER_OP_CPU_KERNEL(cast, ops::CastOpKernel<CPU, float>,
                       ops::CastOpKernel<CPU, double>,
                       ops::CastOpKernel<CPU, int>,
                       ops::CastOpKernel<CPU, int64_t>);
