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
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/operators/math/compound_functors.h"
#include "paddle/fluid/operators/math/functors.h"

namespace math = paddle::operators::math;

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T, typename BinaryFunctor,
          typename UnaryFunctor>
static void RunBinaryCompoundFunctor(
    const framework::ExecutionContext &ctx, const BinaryFunctor &binary_functor,
    const UnaryFunctor &unary_functor, const framework::Tensor &in_x,
    const framework::Tensor &in_y, std::vector<framework::Tensor *> *outputs) {
  // Z = Binary(X, Unary(Y))
  // intermediate_out = Unary(Y)
  // out = Binary(X, Unary(Y))
  // In this case, the shape of intermediate_out and out are different.
  math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor> compound_func(
      binary_functor, unary_functor);
  int axis = ctx.Attr<int>("axis");
  if (ctx.Attr<bool>("keep_intermediate_value")) {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
        true /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
        false /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  }
}

template <typename DeviceContext, typename T, typename UnaryFunctor,
          typename BinaryFunctor>
static void RunUnaryCompoundFunctors(
    const framework::ExecutionContext &ctx, const UnaryFunctor &unary_functor,
    const BinaryFunctor &binary_functor, const framework::Tensor &in_x,
    const framework::Tensor &in_y, std::vector<framework::Tensor *> *outputs) {
  // Z = Unary(Binary(X, Y))
  // intermediate_out = Binary(X, Y)
  // out = Unary(Binary(X, Y))
  // In this case, the shape of intermediate_out and out are the same.
  int axis = ctx.Attr<int>("axis");

  math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor> compound_func(
      unary_functor, binary_functor);

  if (ctx.Attr<bool>("keep_intermediate_value")) {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        true /*KeepIntermediateValue*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        false /*KeepIntermediateValue*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  }
}

template <typename DeviceContext, typename T, typename BinaryGradFunctor,
          typename UnaryFunctor, typename UnaryGradFunctor>
static void RunBinaryCompoundGradFunctors(
    const framework::ExecutionContext &ctx,
    const BinaryGradFunctor &binary_grad_functor,
    const UnaryFunctor &unary_functor,
    const UnaryGradFunctor &unary_grad_functor, const framework::Tensor *in_x,
    const framework::Tensor *in_y, const framework::Tensor *in_out,
    const framework::Tensor *in_intermediate_out,
    const framework::Tensor *in_out_grad, framework::Tensor *x_grad,
    framework::Tensor *y_grad) {
  // Z = Binary(X, Unary(Y))
  int axis = ctx.Attr<int>("axis");

  using BinaryCompoundDxFunctor =
      math::BinaryCompoundGradDxFunctor<T, BinaryGradFunctor, UnaryFunctor>;
  using BinaryCompoundDyFunctor =
      math::BinaryCompoundGradDyFunctor<T, BinaryGradFunctor, UnaryFunctor,
                                        UnaryGradFunctor>;

  if (in_intermediate_out) {
    FusedElemwiseAndActGradComputeEx<
        DeviceContext, T, BinaryCompoundDxFunctor, BinaryCompoundDyFunctor,
        true /*UseIntermediateOut*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
        BinaryCompoundDyFunctor(binary_grad_functor, unary_functor,
                                unary_grad_functor));
  } else {
    FusedElemwiseAndActGradComputeEx<
        DeviceContext, T, BinaryCompoundDxFunctor, BinaryCompoundDyFunctor,
        false /*UseIntermediateOut*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
        BinaryCompoundDyFunctor(binary_grad_functor, unary_functor,
                                unary_grad_functor));
  }
}

template <typename DeviceContext, typename T, typename UnaryGradFunctor,
          typename BinaryFunctor, typename BinaryGradFunctor,
          bool Recomputation = true>
static void RunUnaryCompoundGradFunctors(
    const framework::ExecutionContext &ctx,
    const UnaryGradFunctor &unary_grad_functor,
    const BinaryFunctor &binary_functor,
    const BinaryGradFunctor &binary_grad_functor, const framework::Tensor *in_x,
    const framework::Tensor *in_y, const framework::Tensor *in_out,
    const framework::Tensor *in_intermediate_out,
    const framework::Tensor *in_out_grad, framework::Tensor *x_grad,
    framework::Tensor *y_grad) {
  // Z = Unary(Binary(X, Y))
  int axis = ctx.Attr<int>("axis");

  using UnaryCompoundDxFunctor =
      math::UnaryCompoundGradDxFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor, Recomputation>;
  using UnaryCompoundDyFunctor =
      math::UnaryCompoundGradDyFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor, Recomputation>;

  if (in_intermediate_out) {
    FusedElemwiseAndActGradComputeEx<
        DeviceContext, T, UnaryCompoundDxFunctor, UnaryCompoundDyFunctor,
        true /*UseIntermediateOut*/, true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, UnaryCompoundDxFunctor(unary_grad_functor, binary_functor,
                                       binary_grad_functor),
        UnaryCompoundDyFunctor(unary_grad_functor, binary_functor,
                               binary_grad_functor));
  } else {
    FusedElemwiseAndActGradComputeEx<DeviceContext, T, UnaryCompoundDxFunctor,
                                     UnaryCompoundDyFunctor,
                                     false /*UseIntermediateOut*/,
                                     true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, UnaryCompoundDxFunctor(unary_grad_functor, binary_functor,
                                       binary_grad_functor),
        UnaryCompoundDyFunctor(unary_grad_functor, binary_functor,
                               binary_grad_functor));
  }
}

template <typename DeviceContext, typename T>
static void RunFunctors(const framework::ExecutionContext &ctx,
                        const framework::Tensor &in_x,
                        const framework::Tensor &in_y,
                        std::vector<framework::Tensor *> *outputs) {
  auto &functors = ctx.Attr<std::vector<std::string>>("functor_list");

  // TODO(zcd): The following code can be refined.
  auto funcs_str = functors[0] + "," + functors[1];
  if (funcs_str == "elementwise_add,scale") {
    // Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ScaleFunctor<T>>(
        ctx, math::AddFunctor<T>(), math::ScaleFunctor<T>(scale), in_x, in_y,
        outputs);
  } else if (funcs_str == "scale,elementwise_add") {
    // Z = Unary(Binary(X, Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunUnaryCompoundFunctors<DeviceContext, T, math::ScaleFunctor<T>,
                             math::AddFunctor<T>>(
        ctx, math::ScaleFunctor<T>(scale), math::AddFunctor<T>(), in_x, in_y,
        outputs);
  } else if (funcs_str == "elementwise_add,relu") {
    // Z = Binary(X, Unary(Y))
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ReluFunctor<T>>(ctx, math::AddFunctor<T>(),
                                                   math::ReluFunctor<T>(), in_x,
                                                   in_y, outputs);
  } else if (funcs_str == "relu,elementwise_add") {
    // Z = Unary(Binary(X, Y))
    RunUnaryCompoundFunctors<DeviceContext, T, math::ReluFunctor<T>,
                             math::AddFunctor<T>>(ctx, math::ReluFunctor<T>(),
                                                  math::AddFunctor<T>(), in_x,
                                                  in_y, outputs);
  } else if (funcs_str == "elementwise_mul,scale") {
    // Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundFunctor<DeviceContext, T, math::MulFunctor<T>,
                             math::ScaleFunctor<T>>(
        ctx, math::MulFunctor<T>(), math::ScaleFunctor<T>(scale), in_x, in_y,
        outputs);
  } else {
    PADDLE_THROW("%s has not been implemented.", funcs_str);
  }
}

template <typename DeviceContext, typename T, bool ReComputation>
static void RunGradFunctors(const framework::ExecutionContext &ctx,
                            const framework::Tensor *in_x,
                            const framework::Tensor *in_y,
                            const framework::Tensor *in_out,
                            const framework::Tensor *in_intermediate_out,
                            const framework::Tensor *in_out_grad,
                            framework::Tensor *x_grad,
                            framework::Tensor *y_grad) {
  auto &functors = ctx.Attr<std::vector<std::string>>("functor_list");
  auto funcs_str = functors[0] + "," + functors[1];

  // TODO(zcd): The following code can be refined. for example, use registrition
  if (funcs_str == "elementwise_add_grad,scale_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                  math::ScaleFunctor<T>,
                                  math::ScaleGradFunctor<T>>(
        ctx, math::AddGradFunctor<T>(), math::ScaleFunctor<T>(scale),
        math::ScaleGradFunctor<T>(scale), in_x, in_y, in_out,
        in_intermediate_out, in_out_grad, x_grad, y_grad);
  } else if (funcs_str == "scale_grad,elementwise_add_grad") {
    // The backward of Z = Unary(Binary(X, Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunUnaryCompoundGradFunctors<DeviceContext, T, math::ScaleGradFunctor<T>,
                                 math::AddFunctor<T>, math::AddGradFunctor<T>,
                                 ReComputation /*Recomputation*/>(
        ctx, math::ScaleGradFunctor<T>(scale), math::AddFunctor<T>(),
        math::AddGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
        in_out_grad, x_grad, y_grad);
  } else if (funcs_str == "elementwise_add_grad,relu_grad") {
    RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                  math::ReluFunctor<T>,
                                  math::ReluGradFunctor<T>>(
        ctx, math::AddGradFunctor<T>(), math::ReluFunctor<T>(),
        math::ReluGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
        in_out_grad, x_grad, y_grad);
  } else if (funcs_str == "relu_grad,elementwise_add_grad") {
    RunUnaryCompoundGradFunctors<DeviceContext, T, math::ReluGradFunctor<T>,
                                 math::AddFunctor<T>, math::AddGradFunctor<T>,
                                 ReComputation /*Recomputation*/>(
        ctx, math::ReluGradFunctor<T>(), math::AddFunctor<T>(),
        math::AddGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
        in_out_grad, x_grad, y_grad);
  } else if (funcs_str == "elementwise_mul_grad,scale_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundGradFunctors<DeviceContext, T, math::MulGradFunctor<T>,
                                  math::ScaleFunctor<T>,
                                  math::ScaleGradFunctor<T>>(
        ctx, math::MulGradFunctor<T>(), math::ScaleFunctor<T>(scale),
        math::ScaleGradFunctor<T>(scale), in_x, in_y, in_out,
        in_intermediate_out, in_out_grad, x_grad, y_grad);
  } else {
    PADDLE_THROW("%s has not been implemented.", funcs_str);
  }
}

template <typename DeviceContext, typename T>
class FusedElemwiseActivationKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &in_x = detail::Ref(ctx.Input<framework::Tensor>("X"),
                             "Cannot get input tensor %s, variable name = %s",
                             "X", ctx.op().Input("X"));
    auto &in_y = detail::Ref(ctx.Input<framework::Tensor>("Y"),
                             "Cannot get input tensor %s, variable name = %s",
                             "Y", ctx.op().Input("Y"));
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");

    if (!ctx.Attr<bool>("keep_intermediate_value")) {
      outputs.push_back(nullptr);
    }

    RunFunctors<DeviceContext, T>(ctx, in_x, in_y, &outputs);
  }
};

template <typename DeviceContext, typename T>
class FusedElemwiseActivationGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.Input<framework::Tensor>("X");
    auto y = ctx.Input<framework::Tensor>("Y");

    auto &in_outs = ctx.MultiInput<framework::Tensor>("Out");
    auto &in_outs_grad =
        ctx.MultiInput<framework::Tensor>(framework::GradVarName("Out"));

    framework::Tensor *x_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    framework::Tensor *y_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

    if (ctx.Attr<bool>("recomputation")) {
      PADDLE_ENFORCE(
          x != nullptr,
          "The recomputation is opened, so Input(X) should not be absent.");
      PADDLE_ENFORCE(
          y != nullptr,
          "The recomputation is opened, so Input(Y) should not be absent.");
    } else {
      PADDLE_ENFORCE_GE(in_outs.size(), 1,
                        "The recomputation is disabled,"
                        "so the Input('Out') should not be empty.");
    }

    framework::Tensor *in_x, *in_y;
    framework::Tensor *in_out, *in_intermediate_out;
    // If functor_list contains elementwise_add, the backward doesn't use
    // in_x,in_y and in_outs.
    auto functor_list = ctx.Attr<std::vector<std::string>>("functor_list");
    if (functor_list[0] == "elementwise_add" ||
        functor_list[1] == "elementwise_add") {
      in_x = const_cast<framework::Tensor *>(in_outs_grad[0]);
      in_y = const_cast<framework::Tensor *>(in_outs_grad[0]);
      in_out = const_cast<framework::Tensor *>(in_outs_grad[0]);
    } else {
      in_x = const_cast<framework::Tensor *>(x);
      in_y = const_cast<framework::Tensor *>(y);
      in_out = const_cast<framework::Tensor *>(in_outs[0]);
    }

    if (ctx.Attr<bool>("keep_intermediate_value")) {
      PADDLE_ENFORCE_EQ(in_outs.size(), 2,
                        "The option of 'keep_intermediate_value' is opened, "
                        "so the number of 'Out' should be two.");
      in_intermediate_out = const_cast<framework::Tensor *>(in_outs[1]);
    } else {
      in_intermediate_out = nullptr;
    }

    if (ctx.Attr<bool>("recomputation")) {
      RunGradFunctors<DeviceContext, T, true /*Recomputation*/>(
          ctx, in_x, in_y, in_out, in_intermediate_out, in_outs_grad[0], x_grad,
          y_grad);
    } else {
      RunGradFunctors<DeviceContext, T, false /*Recomputation*/>(
          ctx, in_x, in_y, in_out, in_intermediate_out, in_outs_grad[0], x_grad,
          y_grad);
    }
  }
};
}  // namespace operators
}  // namespace paddle
