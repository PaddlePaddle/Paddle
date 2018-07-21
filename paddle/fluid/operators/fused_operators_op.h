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
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FusedOperatorsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override{};

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

for example,

add;scale,k
div;relu

)DOC");
  };
};

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
using math = paddle::operators::math;

template <typename DeviceContext, typename T>
class FusedOperatorsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *in_y = ctx.Input<Tensor>("Y");
    Tensor *output = ctx.Output<Tensor>("Out");

    auto out_data_ptr = output->mutable_data<T>(ctx.GetPlace());

    std::vector<std::string> functors =
        ctx.Attr<std::vector<std::string>>("functor_list");

    int64_t numel = in_x->numel();

    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        static_cast<size_t>(numel));

    int mode = FuncitonMode(functors);

    if (mode == 1) {
      T scale = 0.1;
      math::BinaryCompoundFunctor<T, math::AddFunctor<T>, math::ScaleFunctor<T>>
          binary_compound_functor(
              in_x->data<T>(), in_y->data<T>(), math::AddFunctor<T>(),
              math::ScaleFunctor<T>(scale), numel, out_data_ptr);

      for_range(binary_compound_functor);
    } else {
      T scale = 0.1;
      math::UnaryCompoundFunctor<T, math::ScaleFunctor<T>, math::AddFunctor<T>>
          unary_compound_functor(in_x->data<T>(), in_y->data<T>(),
                                 math::ScaleFunctor<T>(scale),
                                 math::AddFunctor<T>(), numel, out_data_ptr);

      for_range(unary_compound_functor);
    }
  }

  int FuncitonMode(const std::vector<std::string> &functors) const {
    std::unordered_set<std::string> unary_fun = {"scale", "relu"};
    std::unordered_set<std::string> binary_fun = {"add"};
    std::string unary_fun_str;
    int flag = -1;
    if (binary_fun.count(functors[0])) {
      unary_fun_str = functors[1];
      flag = 2;
    } else if (binary_fun.count(functors[1])) {
      unary_fun_str = functors[0];
      flag = 1;
    } else {
      PADDLE_THROW("functor list is invalid.");
    }
    size_t pos = unary_fun_str.find(",");
    PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str.substr(0, pos)), 1);
    return flag;
  }
};

template <typename DeviceContext, typename T>
class FusedOperatorsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *in_y = ctx.Input<Tensor>("Y");
    const Tensor *in_out = ctx.Input<Tensor>("Out");
    const Tensor *in_out_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));

    Tensor *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor *y_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto x_grad_data_ptr = x_grad->mutable_data<T>(ctx.GetPlace());
    auto y_grad_data_ptr = y_grad->mutable_data<T>(ctx.GetPlace());

    std::vector<std::string> functors =
        ctx.Attr<std::vector<std::string>>("functor_list");

    PADDLE_ENFORCE_EQ(functors.size(), 2);

    int64_t numel = in_x->numel();
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        static_cast<size_t>(numel));

    int mode = FuncitonMode(functors);  // TODO(zcd): get function mode

    if (mode == 1) {
      T scale = 0.1;
      math::UnaryCompoundGradFunctor<T, math::ScaleGradFunctor<T>,
                                     math::AddFunctor<T>,
                                     math::AddGradFunctor<T>>
          unary_compound_functor(
              in_x->data<T>(), in_y->data<T>(), in_out->data<T>(),
              in_out_grad->data<T>(), numel, math::ScaleGradFunctor<T>(scale),
              math::AddFunctor<T>(), math::AddGradFunctor<T>(), x_grad_data_ptr,
              y_grad_data_ptr);

      for_range(unary_compound_functor);
    } else {
      T scale = 0.1;
      math::BinaryCompoundGradFunctor<T, math::AddFunctor<T>,
                                      math::ScaleFunctor<T>,
                                      math::ScaleGradFunctor<T>>
          binary_compound_functor(
              in_x->data<T>(), in_y->data<T>(), in_out->data<T>(),
              in_out_grad->data<T>(), numel, math::AddFunctor<T>(),
              math::ScaleFunctor<T>(scale), math::ScaleGradFunctor<T>(scale),
              x_grad_data_ptr, y_grad_data_ptr);

      for_range(binary_compound_functor);
    }
  }

  int FuncitonMode(const std::vector<std::string> &functors) const {
    std::unordered_set<std::string> unary_fun = {"scale", "relu"};
    std::unordered_set<std::string> binary_fun = {"add"};
    std::string unary_fun_str;
    int flag = -1;
    if (binary_fun.count(functors[0])) {
      unary_fun_str = functors[1];
      flag = 2;
    } else if (binary_fun.count(functors[1])) {
      unary_fun_str = functors[0];
      flag = 1;
    } else {
      PADDLE_THROW("functor list is invalid.");
    }
    size_t pos = unary_fun_str.find(",");
    PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str.substr(0, pos)), 1);
    return flag;
  }
};

class FusedOperatorsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

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
