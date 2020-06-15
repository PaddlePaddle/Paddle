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
#include <math.h>
#include <type_traits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
struct EqualFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a, const T& b) const {
    if (std::is_floating_point<T>::value) {
      // This branch will be optimized while compiling if T is integer. It is
      // safe to cast a and b to double.
      return fabs(static_cast<double>(a - b)) < 1e-8;
    } else {
      return (a == b);
    }
  }
};

template <typename T>
struct NotEqualFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a, const T& b) const {
    return !EqualFunctor<T>()(a, b);
  }
};

#define DEFINE_COMPARE_FUNCTOR(Func, expr)                     \
  template <typename T>                                        \
  struct Func##Functor {                                       \
    using ELEM_TYPE = T;                                       \
    HOSTDEVICE bool operator()(const T& a, const T& b) const { \
      return a expr b;                                         \
    }                                                          \
  };                                                           \
  template <typename T>                                        \
  struct Inverse##Func##Functor {                              \
    using ELEM_TYPE = T;                                       \
    HOSTDEVICE bool operator()(const T& a, const T& b) const { \
      return b expr a;                                         \
    }                                                          \
  };

DEFINE_COMPARE_FUNCTOR(LessThan, <)
DEFINE_COMPARE_FUNCTOR(LessEqual, <=)
DEFINE_COMPARE_FUNCTOR(GreaterThan, >)
DEFINE_COMPARE_FUNCTOR(GreaterEqual, >=)
#undef DEFINE_COMPARE_FUNCTOR

template <typename DeviceContext, typename Functor,
          typename InverseFunctor = Functor>
class CompareOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    int axis = context.Attr<int>("axis");

    if (x->dims().size() >= y->dims().size()) {
      ElementwiseComputeEx<Functor, DeviceContext, T, bool>(context, x, y, axis,
                                                            Functor(), z);
    } else {
      ElementwiseComputeEx<InverseFunctor, DeviceContext, T, bool>(
          context, x, y, axis, InverseFunctor(), z);
    }
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_KERNEL(op_type, dev, functor, inversefunctor)       \
  REGISTER_OP_##dev##_KERNEL(op_type,                                        \
                             ::paddle::operators::CompareOpKernel<           \
                                 ::paddle::platform::dev##DeviceContext,     \
                                 functor<int>, inversefunctor<int>>,         \
                             ::paddle::operators::CompareOpKernel<           \
                                 ::paddle::platform::dev##DeviceContext,     \
                                 functor<int64_t>, inversefunctor<int64_t>>, \
                             ::paddle::operators::CompareOpKernel<           \
                                 ::paddle::platform::dev##DeviceContext,     \
                                 functor<float>, inversefunctor<float>>,     \
                             ::paddle::operators::CompareOpKernel<           \
                                 ::paddle::platform::dev##DeviceContext,     \
                                 functor<double>, inversefunctor<double>>);

#define REGISTER_COMPARE_EQUAL_KERNEL(op_type, dev, functor)              \
  REGISTER_OP_##dev##_KERNEL(                                             \
      op_type, ::paddle::operators::CompareOpKernel<                      \
                   ::paddle::platform::dev##DeviceContext, functor<int>>, \
      ::paddle::operators::CompareOpKernel<                               \
          ::paddle::platform::dev##DeviceContext, functor<int64_t>>,      \
      ::paddle::operators::CompareOpKernel<                               \
          ::paddle::platform::dev##DeviceContext, functor<float>>,        \
      ::paddle::operators::CompareOpKernel<                               \
          ::paddle::platform::dev##DeviceContext, functor<double>>,       \
      ::paddle::operators::CompareOpKernel<                               \
          ::paddle::platform::dev##DeviceContext, functor<bool>>);
