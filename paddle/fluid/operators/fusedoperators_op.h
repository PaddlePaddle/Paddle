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

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/operators/math/functors.h"

namespace math = paddle::operators::math;

namespace paddle {
namespace operators {

class FusedOperatorsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

class FusedOperatorsMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class FusedOperatorsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

static bool IsUnaryCompound(const std::vector<std::string> &functors) {
  std::unordered_set<std::string> unary_fun = {"scale", "relu"};
  std::unordered_set<std::string> binary_fun = {"add", "sub"};

  std::string unary_fun_str;
  bool unary_compound = false;
  if (binary_fun.count(functors[0])) {
    unary_fun_str = functors[1];
  } else if (binary_fun.count(functors[1])) {
    unary_fun_str = functors[0];
    unary_compound = true;
  } else {
    PADDLE_THROW("functor list is invalid.");
  }
  size_t pos = unary_fun_str.find(",");
  PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str.substr(0, pos)), 1);
  return unary_compound;
}

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FusedOperatorsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *in_y = ctx.Input<Tensor>("Y");
    Tensor *output = ctx.Output<Tensor>("Out");

    int axis = ctx.Attr<int>("axis");
    std::vector<std::string> functors =
        ctx.Attr<std::vector<std::string>>("functor_list");

    bool unary_compound = IsUnaryCompound(functors);
    T scale;

    auto unary_fun_str = unary_compound ? functors[0] : functors[1];

    size_t pos = unary_fun_str.find(",");

    auto unary_fun_name = unary_fun_str.substr(0, pos);

    // TODO(zcd): The following code can be refined
    // unary function is scale
    if (unary_fun_name == "scale") {
      std::string scale_str =
          unary_fun_str.substr(pos + 1, unary_fun_str.size());
      try {
        scale = std::stof(scale_str);
      } catch (...) {
        PADDLE_THROW("%s cannot convert to float.", scale_str);
      }

      if (unary_compound) {
        using UnaryCompoundFunctor =
            math::UnaryCompoundFunctor<T, math::ScaleFunctor<T>,
                                       math::AddFunctor<T>>;

        ElementwiseComputeEx<UnaryCompoundFunctor, DeviceContext, T>(
            ctx, in_x, in_y, axis,
            UnaryCompoundFunctor(math::ScaleFunctor<T>(scale),
                                 math::AddFunctor<T>()),
            output);

      } else {
        using BinaryCompoundFunctor =
            math::BinaryCompoundFunctor<T, math::AddFunctor<T>,
                                        math::ScaleFunctor<T>>;

        ElementwiseComputeEx<BinaryCompoundFunctor, DeviceContext, T>(
            ctx, in_x, in_y, axis,
            BinaryCompoundFunctor(math::AddFunctor<T>(),
                                  math::ScaleFunctor<T>(scale)),
            output);
      }
    } else if (unary_fun_name == "relu") {
      if (unary_compound) {
        using UnaryCompoundFunctor =
            math::UnaryCompoundFunctor<T, math::ReluFunctor<T>,
                                       math::AddFunctor<T>>;

        ElementwiseComputeEx<UnaryCompoundFunctor, DeviceContext, T>(
            ctx, in_x, in_y, axis,
            UnaryCompoundFunctor(math::ReluFunctor<T>(), math::AddFunctor<T>()),
            output);

      } else {
        using BinaryCompoundFunctor =
            math::BinaryCompoundFunctor<T, math::AddFunctor<T>,
                                        math::ReluFunctor<T>>;

        ElementwiseComputeEx<BinaryCompoundFunctor, DeviceContext, T>(
            ctx, in_x, in_y, axis,
            BinaryCompoundFunctor(math::AddFunctor<T>(),
                                  math::ReluFunctor<T>()),
            output);
      }
    } else {
      PADDLE_THROW("%s has not been implemented.", unary_fun_name);
    }
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

    int axis = ctx.Attr<int>("axis");
    std::vector<std::string> functors =
        ctx.Attr<std::vector<std::string>>("functor_list");

    PADDLE_ENFORCE_EQ(functors.size(), 2);

    int64_t numel = in_x->numel();
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        static_cast<size_t>(numel));

    bool unary_compound = IsUnaryCompound(functors);
    T scale;

    auto unary_fun_str = unary_compound ? functors[0] : functors[1];
    size_t pos = unary_fun_str.find(",");
    auto unary_fun_name = unary_fun_str.substr(0, pos);

    // TODO(zcd): The following code can be refined
    if (unary_fun_name == "scale") {
      std::string scale_str =
          unary_fun_str.substr(pos + 1, unary_fun_str.size());
      try {
        scale = std::stof(scale_str);
      } catch (...) {
        PADDLE_THROW("%s cannot convert to float.", scale_str);
      }

      if (unary_compound) {
        using UnaryCompoundDxFunctor =
            math::UnaryCompoundGradDxFunctor<T, math::ScaleGradFunctor<T>,
                                             math::AddFunctor<T>,
                                             math::AddGradFunctor<T>>;
        using UnaryCompoundDyFunctor =
            math::UnaryCompoundGradDyFunctor<T, math::ScaleGradFunctor<T>,
                                             math::AddFunctor<T>,
                                             math::AddGradFunctor<T>>;

        ElemwiseGradCompute<DeviceContext, T, UnaryCompoundDxFunctor,
                            UnaryCompoundDyFunctor>(
            ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
            UnaryCompoundDxFunctor(math::ScaleGradFunctor<T>(scale),
                                   math::AddFunctor<T>(),
                                   math::AddGradFunctor<T>()),
            UnaryCompoundDyFunctor(math::ScaleGradFunctor<T>(scale),
                                   math::AddFunctor<T>(),
                                   math::AddGradFunctor<T>()));
      } else {
        using BinaryCompoundDxFunctor =
            math::BinaryCompoundGradDxFunctor<T, math::AddGradFunctor<T>,
                                              math::ScaleFunctor<T>>;
        using BinaryCompoundDyFunctor =
            math::BinaryCompoundGradDyFunctor<T, math::AddGradFunctor<T>,
                                              math::ScaleFunctor<T>,
                                              math::ScaleGradFunctor<T>>;

        ElemwiseGradCompute<DeviceContext, T, BinaryCompoundDxFunctor,
                            BinaryCompoundDyFunctor>(
            ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
            BinaryCompoundDxFunctor(math::AddGradFunctor<T>(),
                                    math::ScaleFunctor<T>(scale)),
            BinaryCompoundDyFunctor(math::AddGradFunctor<T>(),
                                    math::ScaleFunctor<T>(scale),
                                    math::ScaleGradFunctor<T>(scale)));
      }
    } else if (unary_fun_name == "relu") {
      if (unary_compound) {
        using UnaryCompoundDxFunctor =
            math::UnaryCompoundGradDxFunctor<T, math::ReluGradFunctor<T>,
                                             math::AddFunctor<T>,
                                             math::AddGradFunctor<T>>;
        using UnaryCompoundDyFunctor =
            math::UnaryCompoundGradDyFunctor<T, math::ReluGradFunctor<T>,
                                             math::AddFunctor<T>,
                                             math::AddGradFunctor<T>>;

        ElemwiseGradCompute<DeviceContext, T, UnaryCompoundDxFunctor,
                            UnaryCompoundDyFunctor>(
            ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
            UnaryCompoundDxFunctor(math::ReluGradFunctor<T>(),
                                   math::AddFunctor<T>(),
                                   math::AddGradFunctor<T>()),
            UnaryCompoundDyFunctor(math::ReluGradFunctor<T>(),
                                   math::AddFunctor<T>(),
                                   math::AddGradFunctor<T>()));
      } else {
        using BinaryCompoundDxFunctor =
            math::BinaryCompoundGradDxFunctor<T, math::AddGradFunctor<T>,
                                              math::ReluFunctor<T>>;
        using BinaryCompoundDyFunctor =
            math::BinaryCompoundGradDyFunctor<T, math::AddGradFunctor<T>,
                                              math::ReluFunctor<T>,
                                              math::ReluGradFunctor<T>>;

        ElemwiseGradCompute<DeviceContext, T, BinaryCompoundDxFunctor,
                            BinaryCompoundDyFunctor>(
            ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
            BinaryCompoundDxFunctor(math::AddGradFunctor<T>(),
                                    math::ReluFunctor<T>()),
            BinaryCompoundDyFunctor(math::AddGradFunctor<T>(),
                                    math::ReluFunctor<T>(),
                                    math::ReluGradFunctor<T>()));
      }
    } else {
      PADDLE_THROW("%s has not been implemented.", unary_fun_name);
    }
  }
};

}  // namespace operators
}  // namespace paddle
