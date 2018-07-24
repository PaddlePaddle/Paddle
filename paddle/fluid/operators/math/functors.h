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

#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

// AddFunctor
template <typename T>
struct AddFunctor {
  // out = x + y;
  inline HOSTDEVICE T operator()(const T x, const T y) const { return x + y; }
};

template <typename T>
struct AddGradFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y) const { return 1; }

  inline HOSTDEVICE T operator()(const T x, const T y, const T out) const {
    return 1;
  }
};

// ScaleFunctor
template <typename T>
struct ScaleFunctor {
  explicit ScaleFunctor(const T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T operator()(const T ele) const { return ele * coeff_; }

 private:
  T coeff_;
};

template <typename T>
struct ScaleGradFunctor {
  explicit ScaleGradFunctor(T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T operator()(const T x) const { return coeff_; }

  inline HOSTDEVICE T operator()(const T x, const T out) const {
    return coeff_;
  }

 private:
  T coeff_;
};

// ReluFunctor
template <typename T>
struct ReluFunctor {
  inline HOSTDEVICE T operator()(const T x) const { return x * (x > 0); }
};

template <typename T>
struct ReluGradFunctor {
  inline HOSTDEVICE T operator()(const T x) const { return x > 0 ? 1 : 0; }

  inline HOSTDEVICE T operator()(const T x, const T out) const {
    return x > 0 ? 1 : 0;
  }
};

// for example: z = x + scale * y
// fun1: t = scale * y
// fun2: z = x + t
template <typename T, typename BinaryFun, typename UnaryFun>
struct BinaryCompoundFunctor {
  BinaryCompoundFunctor(const BinaryFun &binary_fun, const UnaryFun &unary_fun)
      : binary_fun_(binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y) const {
    return binary_fun_(x, unary_fun_(y));
  }

 private:
  BinaryFun binary_fun_;
  UnaryFun unary_fun_;
};

// for example: z = x + scale * y
// fun1: z = (x + t)'
// fun2: t = scale * y
template <typename T, typename DBinaryFun, typename UnaryFun>
struct BinaryCompoundGradDxFunctor {
  BinaryCompoundGradDxFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    //    return dout * d_binary_fun_(x, unary_fun_(y), out);
    return dout * d_binary_fun_(x, unary_fun_(y));
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

// for example: z = x + scale * y
// fun1: z = (x + t)'
// fun2: t = scale * y
// fun3: t = (scale * x)'
template <typename T, typename DBinaryFun, typename UnaryFun,
          typename DUnaryFun>
struct BinaryCompoundGradDyFunctor {
  BinaryCompoundGradDyFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun,
                              const DUnaryFun &d_unary_fun)
      : d_binary_fun_(d_binary_fun),
        unary_fun_(unary_fun),
        d_unary_fun_(d_unary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    return dout * d_binary_fun_(unary_fun_(y), x) * d_unary_fun_(y);
    //    return dout * d_binary_fun_(unary_fun_(y), x, out) * d_unary_fun_(y);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
  DUnaryFun d_unary_fun_;
};

// for example: z = scale * (x + y)
// fun1: t = scale * x
// fun2: t = x + y
template <typename T, typename UnaryFun, typename BinaryFun>
struct UnaryCompoundFunctor {
  UnaryCompoundFunctor(const UnaryFun &unary_fun, const BinaryFun &binary_fun)
      : unary_fun_(unary_fun), binary_fun_(binary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y) const {
    return unary_fun_(binary_fun_(x, y));
  }

 private:
  UnaryFun unary_fun_;
  BinaryFun binary_fun_;
};

// for example: z = scale * (x + y)
// fun1: t = (scale * x)'
// fun2: t = x + y
// fun3: t = (x + y)'
template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun>
struct UnaryCompoundGradDxFunctor {
  UnaryCompoundGradDxFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    //    auto base = dout * d_unary_fun_(binary_fun_(x, y), out);
    auto base = dout * d_unary_fun_(binary_fun_(x, y));
    return base * d_binary_fun_(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

// for example: z = scale * (x + y)
// fun1: t = (scale * x)'
// fun2: t = x + y
// fun3: t = (x + y)'
template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun>
struct UnaryCompoundGradDyFunctor {
  UnaryCompoundGradDyFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    //    auto base = dout * d_unary_fun_(binary_fun_(x, y), out);
    auto base = dout * d_unary_fun_(binary_fun_(x, y));
    return base * d_binary_fun_(y, x);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

using Tensor = framework::Tensor;

static bool ValidCheck(const std::string &functors) {
  std::unordered_set<std::string> unary_fun = {"scale", "relu"};
  std::unordered_set<std::string> binary_fun = {"add"};

  size_t pos = functors.find(",");
  auto func_1 = functors.substr(0, pos);
  auto func_2 = functors.substr(pos + 1, functors.size());
  std::string unary_fun_str;
  if (binary_fun.count(func_1)) {
    unary_fun_str = func_2;
  } else if (binary_fun.count(func_2)) {
    unary_fun_str = func_1;
  } else {
    PADDLE_THROW("%s and %s are not included in fused_list.", func_1, func_2);
  }
  PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str), 1,
                    "%s is not included in fused_list.", unary_fun_str);
  return true;
}

template <typename DeviceContext, typename T, typename BinaryFunctor,
          typename UnaryFunctor>
static void RunBinaryCompoundFunctor(const framework::ExecutionContext &ctx,
                                     const BinaryFunctor binary_functor,
                                     const UnaryFunctor unary_functor,
                                     const Tensor *in_x, const Tensor *in_y,
                                     Tensor *output) {
  int axis = ctx.Attr<int>("axis");

  using BinaryCompoundFunctor =
      math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>;

  ElementwiseComputeEx<BinaryCompoundFunctor, DeviceContext, T>(
      ctx, in_x, in_y, axis,
      BinaryCompoundFunctor(binary_functor, unary_functor), output);
}

template <typename DeviceContext, typename T, typename UnaryFunctor,
          typename BinaryFunctor>
static void RunUnaryCompoundFunctors(const framework::ExecutionContext &ctx,
                                     const UnaryFunctor unary_functor,
                                     const BinaryFunctor binary_functor,
                                     const Tensor *in_x, const Tensor *in_y,
                                     Tensor *output) {
  int axis = ctx.Attr<int>("axis");

  using UnaryCompoundFunctor =
      math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>;

  ElementwiseComputeEx<UnaryCompoundFunctor, DeviceContext, T>(
      ctx, in_x, in_y, axis,
      UnaryCompoundFunctor(unary_functor, binary_functor), output);
}

template <typename DeviceContext, typename T, typename BinaryGradFunctor,
          typename UnaryFunctor, typename UnaryGradFunctor>
static void RunBinaryCompoundGradFunctors(
    const framework::ExecutionContext &ctx,
    const BinaryGradFunctor binary_grad_functor,
    const UnaryFunctor unary_functor, const UnaryGradFunctor unary_grad_functor,
    const Tensor *in_x, const Tensor *in_y, const Tensor *in_out,
    const Tensor *in_out_grad, Tensor *x_grad, Tensor *y_grad) {
  int axis = ctx.Attr<int>("axis");

  using BinaryCompoundDxFunctor =
      math::BinaryCompoundGradDxFunctor<T, BinaryGradFunctor, UnaryFunctor>;
  using BinaryCompoundDyFunctor =
      math::BinaryCompoundGradDyFunctor<T, BinaryGradFunctor, UnaryFunctor,
                                        UnaryGradFunctor>;

  ElemwiseGradCompute<DeviceContext, T, BinaryCompoundDxFunctor,
                      BinaryCompoundDyFunctor>(
      ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
      BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
      BinaryCompoundDyFunctor(binary_grad_functor, unary_functor,
                              unary_grad_functor));
}

template <typename DeviceContext, typename T, typename UnaryGradFunctor,
          typename BinaryFunctor, typename BinaryGradFunctor>
static void RunUnaryCompoundGradFunctors(
    const framework::ExecutionContext &ctx,
    const UnaryGradFunctor unary_grad_functor,
    const BinaryFunctor binary_functor,
    const BinaryGradFunctor binary_grad_functor, const Tensor *in_x,
    const Tensor *in_y, const Tensor *in_out, const Tensor *in_out_grad,
    Tensor *x_grad, Tensor *y_grad) {
  int axis = ctx.Attr<int>("axis");

  using UnaryCompoundDxFunctor =
      math::UnaryCompoundGradDxFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor>;
  using UnaryCompoundDyFunctor =
      math::UnaryCompoundGradDyFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor>;

  ElemwiseGradCompute<DeviceContext, T, UnaryCompoundDxFunctor,
                      UnaryCompoundDyFunctor>(
      ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
      UnaryCompoundDxFunctor(unary_grad_functor, binary_functor,
                             binary_grad_functor),
      UnaryCompoundDyFunctor(unary_grad_functor, binary_functor,
                             binary_grad_functor));
}

template <typename DeviceContext, typename T>
static void RunFunctors(const framework::ExecutionContext &ctx,
                        const std::string &functors, const Tensor *in_x,
                        const Tensor *in_y, Tensor *output) {
  PADDLE_ENFORCE(ValidCheck(functors));

  // TODO(zcd): The following code can be refined.
  // unary function is scale
  if (functors == "add,scale") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ScaleFunctor<T>>(
        ctx, math::AddFunctor<T>(), math::ScaleFunctor<T>(scale), in_x, in_y,
        output);
  } else if (functors == "scale,add") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunUnaryCompoundFunctors<DeviceContext, T, math::ScaleFunctor<T>,
                             math::AddFunctor<T>>(
        ctx, math::ScaleFunctor<T>(scale), math::AddFunctor<T>(), in_x, in_y,
        output);
  } else if (functors == "add,relu") {
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ReluFunctor<T>>(
        ctx, math::AddFunctor<T>(), math::ReluFunctor<T>(), in_x, in_y, output);
  } else if (functors == "relu,add") {
    RunUnaryCompoundFunctors<DeviceContext, T, math::ReluFunctor<T>,
                             math::AddFunctor<T>>(
        ctx, math::ReluFunctor<T>(), math::AddFunctor<T>(), in_x, in_y, output);
  } else {
    PADDLE_THROW("%s has not been implemented.", functors);
  }
}

template <typename DeviceContext, typename T>
static void RunGradFunctors(const framework::ExecutionContext &ctx,
                            const std::string &functors, const Tensor *in_x,
                            const Tensor *in_y, const Tensor *in_out,
                            const Tensor *in_out_grad, Tensor *x_grad,
                            Tensor *y_grad) {
  PADDLE_ENFORCE(ValidCheck(functors));

  // TODO(zcd): The following code can be refined.
  // unary function is scale
  if (functors == "add,scale") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                  math::ScaleFunctor<T>,
                                  math::ScaleGradFunctor<T>>(
        ctx, math::AddGradFunctor<T>(), math::ScaleFunctor<T>(scale),
        math::ScaleGradFunctor<T>(scale), in_x, in_y, in_out, in_out_grad,
        x_grad, y_grad);
  } else if (functors == "scale,add") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunUnaryCompoundGradFunctors<DeviceContext, T, math::ScaleGradFunctor<T>,
                                 math::AddFunctor<T>, math::AddGradFunctor<T>>(
        ctx, math::ScaleGradFunctor<T>(scale), math::AddFunctor<T>(),
        math::AddGradFunctor<T>(), in_x, in_y, in_out, in_out_grad, x_grad,
        y_grad);
  } else if (functors == "add,relu") {
    RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                  math::ReluFunctor<T>,
                                  math::ReluGradFunctor<T>>(
        ctx, math::AddGradFunctor<T>(), math::ReluFunctor<T>(),
        math::ReluGradFunctor<T>(), in_x, in_y, in_out, in_out_grad, x_grad,
        y_grad);
  } else if (functors == "relu,add") {
    RunUnaryCompoundGradFunctors<DeviceContext, T, math::ReluGradFunctor<T>,
                                 math::AddFunctor<T>, math::AddGradFunctor<T>>(
        ctx, math::ReluGradFunctor<T>(), math::AddFunctor<T>(),
        math::AddGradFunctor<T>(), in_x, in_y, in_out, in_out_grad, x_grad,
        y_grad);
  } else {
    PADDLE_THROW("%s has not been implemented.", functors);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
