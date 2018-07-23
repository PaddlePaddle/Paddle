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

template <typename DeviceContext, typename T>
static void RunFunctors(const framework::ExecutionContext &ctx,
                        const std::vector<std::string> &functors,
                        const Tensor *in_x, const Tensor *in_y,
                        Tensor *output) {
  int axis = ctx.Attr<int>("axis");

  bool unary_compound = IsUnaryCompound(functors);
  T scale;

  auto unary_fun_str = unary_compound ? functors[0] : functors[1];

  size_t pos = unary_fun_str.find(",");

  auto unary_fun_name = unary_fun_str.substr(0, pos);

  // TODO(zcd): The following code can be refined
  // unary function is scale
  if (unary_fun_name == "scale") {
    std::string scale_str = unary_fun_str.substr(pos + 1, unary_fun_str.size());
    try {
      scale = stof(scale_str);
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
          BinaryCompoundFunctor(math::AddFunctor<T>(), math::ReluFunctor<T>()),
          output);
    }
  } else {
    PADDLE_THROW("%s has not been implemented.", unary_fun_name);
  }
}

template <typename DeviceContext, typename T>
static void RunGradFunctors(const framework::ExecutionContext &ctx,
                            const std::vector<std::string> &functors,
                            const Tensor *in_x, const Tensor *in_y,
                            const Tensor *in_out, const Tensor *in_out_grad,
                            Tensor *x_grad, Tensor *y_grad) {
  int axis = ctx.Attr<int>("axis");
  bool unary_compound = IsUnaryCompound(functors);
  T scale;

  auto unary_fun_str = unary_compound ? functors[0] : functors[1];
  size_t pos = unary_fun_str.find(",");
  auto unary_fun_name = unary_fun_str.substr(0, pos);

  // TODO(zcd): The following code can be refined
  if (unary_fun_name == "scale") {
    std::string scale_str = unary_fun_str.substr(pos + 1, unary_fun_str.size());
    try {
      scale = stof(scale_str);
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

}  // namespace math
}  // namespace operators
}  // namespace paddle
