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

#pragma once
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/functors.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FusedOperatorsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override{};

 protected:
  //  framework::OpKernelType GetExpectedKernelType(
  //      const framework::ExecutionContext& ctx) const override{};
};

class FusedOperatorsMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Inputs", "(vector<Tensor>)").AsDuplicable();
    AddOutput("Output", "vector<Tensor>");
    AddAttr<std::vector<std::string>>("functor_list", "");

    AddComment(R"DOC(
FusedOperators Operator.
)DOC");
  };
};

template <typename DeviceContext, typename T>
class FusedOperatorsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // parse functor_list
    //
    auto in_vars = context.MultiInputVar("Inputs");
    auto out_var = context.MultiOutputVar("Output");
  }
};

template <typename DeviceContext, typename T>
class FusedOperatorsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};

class FusedOperatorsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  //  framework::OpKernelType GetExpectedKernelType(
  //      const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_operators, ops::FusedOperatorsOp,
                  ops::FusedOperatorsMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fused_operators_grad, ops::FusedOperatorsOpGrad);
