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
#include "paddle/fluid/operators/math/functors.h"

namespace math = paddle::operators::math;

namespace paddle {
namespace operators {

template <typename T, typename BinaryFunctor, typename UnaryFunctor>
struct BinaryCompoundFunctor {
  BinaryCompoundFunctor(const BinaryFunctor func1, const UnaryFunctor func2)
      : func1_(func1), func2_(func2) {}
  // Z = BinaryFunctor(X, UnaryFunctor(Y))

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(x, func2_(y)); }

  inline HOSTDEVICE T GetOut(T x, T y, T intermediat_out) {
    return func1_(x, intermediat_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return func2_(y); }

  BinaryFunctor func1_;
  UnaryFunctor func2_;
};

template <typename T, typename UnaryFunctor, typename BinaryFunctor>
struct UnaryCompoundFunctor {
  UnaryCompoundFunctor(const UnaryFunctor func1, const BinaryFunctor func2)
      : func1_(func1), func2_(func2) {}
  // Z = UnaryFunctor(BinaryFunctor(X, Y))

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(func2_(x, y)); }

  inline HOSTDEVICE T GetOut(T x, T y, T intermediat_out) {
    return func1_(intermediat_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return func2_(x, y); }

  UnaryFunctor func1_;
  BinaryFunctor func2_;
};

// FIXME(zcd): DBinaryFun and DUnaryFun have to method to get
// the dx, one is to use the 'out', and the other is not to use it.
// the former method will save the time of recomputing the
// 'out', but it must occupy the memory to store the 'out'.
// While the later method can avoid occupying this memory,
// but it must recompute the 'out'.
template <typename T, typename DBinaryFun, typename UnaryFun,
          bool Recomputation = true>
struct BinaryCompoundGradDxFunctor {
  BinaryCompoundGradDxFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    if (Recomputation) {
      return dout * d_binary_fun_(x, unary_fun_(y));
    } else {
      return dout * d_binary_fun_(x, unary_fun_(y), out);
    }
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    if (Recomputation) {
      return dout * d_binary_fun_(x, intermediate_out);
    } else {
      return dout * d_binary_fun_(x, intermediate_out, out);
    }
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

template <typename T, typename DBinaryFun, typename UnaryFun,
          typename DUnaryFun, bool Recomputation = true>
struct BinaryCompoundGradDyFunctor {
  BinaryCompoundGradDyFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun,
                              const DUnaryFun &d_unary_fun)
      : d_binary_fun_(d_binary_fun),
        unary_fun_(unary_fun),
        d_unary_fun_(d_unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    if (Recomputation) {
      return dout * d_binary_fun_(unary_fun_(y), x) * d_unary_fun_(y);
    } else {
      return dout * d_binary_fun_(unary_fun_(y), x, out) * d_unary_fun_(y);
    }
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    if (Recomputation) {
      return dout * d_binary_fun_(intermediate_out, x) * d_unary_fun_(y);
    } else {
      return dout * d_binary_fun_(intermediate_out, x, out) * d_unary_fun_(y);
    }
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
  DUnaryFun d_unary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDxFunctor {
  UnaryCompoundGradDxFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_(x, y);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(intermediate_out);
    } else {
      base = dout * d_unary_fun_(intermediate_out, out);
    }
    return base * d_binary_fun_(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDyFunctor {
  UnaryCompoundGradDyFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_(y, x);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(intermediate_out);
    } else {
      base = dout * d_unary_fun_(intermediate_out, out);
    }
    return base * d_binary_fun_(y, x);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

template <typename DeviceContext, typename T, typename BinaryFunctor,
          typename UnaryFunctor>
static void RunBinaryCompoundFunctor(
    const framework::ExecutionContext &ctx, const BinaryFunctor &binary_functor,
    const UnaryFunctor &unary_functor, const framework::Tensor &in_x,
    const framework::Tensor &in_y, std::vector<framework::Tensor *> *outputs) {
  // Z = Binary(X, Unary(Y))
  int axis = ctx.Attr<int>("axis");

  BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor> compound_func(
      binary_functor, unary_functor);

  if (ctx.Attr<bool>("keep_intermediate_value")) {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T, BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
        true /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T, BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
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
  int axis = ctx.Attr<int>("axis");

  UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor> compound_func(
      unary_functor, binary_functor);

  if (ctx.Attr<bool>("keep_intermediate_value")) {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T, UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        true /*KeepIntermediateValue*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T, UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        false /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  }
}

template <typename DeviceContext, typename T, typename BinaryGradFunctor,
          typename UnaryFunctor, typename UnaryGradFunctor,
          bool Recomputation = true>
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
      BinaryCompoundGradDxFunctor<T, BinaryGradFunctor, UnaryFunctor,
                                  Recomputation>;
  using BinaryCompoundDyFunctor =
      BinaryCompoundGradDyFunctor<T, BinaryGradFunctor, UnaryFunctor,
                                  UnaryGradFunctor, Recomputation>;

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
      UnaryCompoundGradDxFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                 BinaryGradFunctor, Recomputation>;
  using UnaryCompoundDyFunctor =
      UnaryCompoundGradDyFunctor<T, UnaryGradFunctor, BinaryFunctor,
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
  } else {
    PADDLE_THROW("%s has not been implemented.", funcs_str);
  }
}

template <typename DeviceContext, typename T>
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

  bool recomputation = ctx.Attr<bool>("recomputation");

  // TODO(zcd): The following code can be refined. for example, use registrition
  if (funcs_str == "elementwise_add_grad,scale_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    if (recomputation) {
      RunBinaryCompoundGradFunctors<
          DeviceContext, T, math::AddGradFunctor<T>, math::ScaleFunctor<T>,
          math::ScaleGradFunctor<T>, true /*Recomputation*/>(
          ctx, math::AddGradFunctor<T>(), math::ScaleFunctor<T>(scale),
          math::ScaleGradFunctor<T>(scale), in_x, in_y, in_out,
          in_intermediate_out, in_out_grad, x_grad, y_grad);
    } else {
      RunBinaryCompoundGradFunctors<
          DeviceContext, T, math::AddGradFunctor<T>, math::ScaleFunctor<T>,
          math::ScaleGradFunctor<T>, false /*Recomputation*/>(
          ctx, math::AddGradFunctor<T>(), math::ScaleFunctor<T>(scale),
          math::ScaleGradFunctor<T>(scale), in_x, in_y, in_out,
          in_intermediate_out, in_out_grad, x_grad, y_grad);
    }
  } else if (funcs_str == "scale_grad,elementwise_add_grad") {
    // The backward of Z = Unary(Binary(X, Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    if (recomputation) {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ScaleGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   true /*Recomputation*/>(
          ctx, math::ScaleGradFunctor<T>(scale), math::AddFunctor<T>(),
          math::AddGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
          in_out_grad, x_grad, y_grad);
    } else {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ScaleGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   false /*Recomputation*/>(
          ctx, math::ScaleGradFunctor<T>(scale), math::AddFunctor<T>(),
          math::AddGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
          in_out_grad, x_grad, y_grad);
    }
  } else if (funcs_str == "elementwise_add_grad,relu_grad") {
    if (recomputation) {
      RunBinaryCompoundGradFunctors<
          DeviceContext, T, math::AddGradFunctor<T>, math::ReluFunctor<T>,
          math::ReluGradFunctor<T>, true /*Recomputation*/>(
          ctx, math::AddGradFunctor<T>(), math::ReluFunctor<T>(),
          math::ReluGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
          in_out_grad, x_grad, y_grad);
    } else {
      RunBinaryCompoundGradFunctors<
          DeviceContext, T, math::AddGradFunctor<T>, math::ReluFunctor<T>,
          math::ReluGradFunctor<T>, false /*Recomputation*/>(
          ctx, math::AddGradFunctor<T>(), math::ReluFunctor<T>(),
          math::ReluGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
          in_out_grad, x_grad, y_grad);
    }
  } else if (funcs_str == "relu_grad,elementwise_add_grad") {
    if (recomputation) {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ReluGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   true /*Recomputation*/>(
          ctx, math::ReluGradFunctor<T>(), math::AddFunctor<T>(),
          math::AddGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
          in_out_grad, x_grad, y_grad);
    } else {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ReluGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   false /*Recomputation*/>(
          ctx, math::ReluGradFunctor<T>(), math::AddFunctor<T>(),
          math::AddGradFunctor<T>(), in_x, in_y, in_out, in_intermediate_out,
          in_out_grad, x_grad, y_grad);
    }
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
    framework::Tensor in_x = *ctx.Input<framework::Tensor>("X");
    framework::Tensor in_y = *ctx.Input<framework::Tensor>("Y");
    framework::Tensor in_out;

    auto &in_outs = ctx.MultiInput<framework::Tensor>("Out");
    auto &in_outs_grad =
        ctx.MultiInput<framework::Tensor>(framework::GradVarName("Out"));

    framework::Tensor *x_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    framework::Tensor *y_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

    // If functor_list contains elementwise_add, the backward doesn't use
    // in_x,in_y and in_outs.
    if (ctx.Attr<std::vector<std::string>>("functor_list")[0] ==
            "elementwise_add" ||
        ctx.Attr<std::vector<std::string>>("functor_list")[1] ==
            "elementwise_add") {
      in_x = *in_outs_grad[0];
      in_y = *in_outs_grad[0];
      in_out = *in_outs_grad[0];
    } else {
      in_out = *in_outs[0];
    }

    if (ctx.Attr<bool>("keep_intermediate_value")) {
      RunGradFunctors<DeviceContext, T>(ctx, &in_x, &in_y, &in_out, in_outs[1],
                                        in_outs_grad[0], x_grad, y_grad);
    } else {
      RunGradFunctors<DeviceContext, T>(ctx, &in_x, &in_y, &in_out, nullptr,
                                        in_outs_grad[0], x_grad, y_grad);
    }
  }
};
}  // namespace operators
}  // namespace paddle
