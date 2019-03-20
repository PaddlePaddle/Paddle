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

#include "paddle/fluid/operators/cast_op.h"
#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

class CastOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of cast op");
    AddOutput("Out", "The output tensor of cast op");
    AddAttr<int>("out_dtype", "output data type");
    AddAttr<int>("in_dtype", "input data type");
    AddComment(R"DOC(
Cast Operator.

This Operator casts the input tensor to another data type and
returns the Output Tensor. It's meaningless if the output dtype equals
the input dtype, but it's fine if you do so.

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
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto grad = new framework::OpDesc();
    grad->SetType("cast");
    grad->SetInput("X", OutputGrad("Out"));
    grad->SetOutput("Out", InputGrad("X"));
    grad->SetAttr("out_dtype", GetAttr("in_dtype"));
    grad->SetAttr("in_dtype", GetAttr("out_dtype"));
    return std::unique_ptr<framework::OpDesc>(grad);
  }
};

class CastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // CastOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
REGISTER_OPERATOR(cast, ops::CastOp, ops::CastOpGradMaker,
                  ops::CastOpInferShape, ops::CastOpProtoMaker);
REGISTER_OP_CPU_KERNEL(cast, ops::CastOpKernel<CPU, float>,
                       ops::CastOpKernel<CPU, double>,
                       ops::CastOpKernel<CPU, int>,
                       ops::CastOpKernel<CPU, int64_t>,
                       ops::CastOpKernel<CPU, bool>,
                       ops::CastOpKernel<CPU, uint8_t>,
                       ops::CastOpKernel<CPU, paddle::platform::float16>);
