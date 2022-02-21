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

#define LOGICAL_BINARY_FUNCTOR(func_name, op)                \
  template <typename T>                                      \
  struct func_name {                                         \
    using ELEMENT_TYPE = T;                                  \
    HOSTDEVICE bool operator()(const T a, const T b) const { \
      return static_cast<bool>(a) op static_cast<bool>(b);   \
    }                                                        \
  };

LOGICAL_BINARY_FUNCTOR(LogicalOrFunctor, ||)
LOGICAL_BINARY_FUNCTOR(LogicalAndFunctor, &&)
LOGICAL_BINARY_FUNCTOR(LogicalXorFunctor, ^)
#undef LOGICAL_BINARY_FUNCTOR

template <typename T>
struct LogicalNotFunctor {
  using ELEMENT_TYPE = T;
  HOSTDEVICE bool operator()(const T a) const { return !a; }
};

template <typename DeviceContext, typename Functor>
class BinaryLogicalOpKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEMENT_TYPE;
    auto* x = context.Input<framework::Tensor>("X");
    auto* y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");
    Functor binary_func;
    ElementwiseComputeEx<Functor, DeviceContext, T, bool>(context, x, y, -1,
                                                          binary_func, out);
  }
};

template <typename DeviceContext, typename Functor>
class UnaryLogicalOpKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEMENT_TYPE;
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    Functor unary_func;
    platform::Transform<DeviceContext> trans;
    trans(context.template device_context<DeviceContext>(), x->data<T>(),
          x->data<T>() + x->numel(),
          out->mutable_data<bool>(context.GetPlace()), unary_func);
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_BINARY_LOGICAL_KERNEL(op_type, dev, functor)              \
  REGISTER_OP_##dev##_KERNEL(                                              \
      op_type, ::paddle::operators::BinaryLogicalOpKernel<                 \
                   ::paddle::platform::dev##DeviceContext, functor<bool>>, \
      ::paddle::operators::BinaryLogicalOpKernel<                          \
          ::paddle::platform::dev##DeviceContext, functor<int8_t>>,        \
      ::paddle::operators::BinaryLogicalOpKernel<                          \
          ::paddle::platform::dev##DeviceContext, functor<int16_t>>,       \
      ::paddle::operators::BinaryLogicalOpKernel<                          \
          ::paddle::platform::dev##DeviceContext, functor<int>>,           \
      ::paddle::operators::BinaryLogicalOpKernel<                          \
          ::paddle::platform::dev##DeviceContext, functor<int64_t>>,       \
      ::paddle::operators::BinaryLogicalOpKernel<                          \
          ::paddle::platform::dev##DeviceContext, functor<float>>,         \
      ::paddle::operators::BinaryLogicalOpKernel<                          \
          ::paddle::platform::dev##DeviceContext, functor<double>>);

#define REGISTER_UNARY_LOGICAL_KERNEL(op_type, dev, functor)               \
  REGISTER_OP_##dev##_KERNEL(                                              \
      op_type, ::paddle::operators::UnaryLogicalOpKernel<                  \
                   ::paddle::platform::dev##DeviceContext, functor<bool>>, \
      ::paddle::operators::UnaryLogicalOpKernel<                           \
          ::paddle::platform::dev##DeviceContext, functor<int8_t>>,        \
      ::paddle::operators::UnaryLogicalOpKernel<                           \
          ::paddle::platform::dev##DeviceContext, functor<int16_t>>,       \
      ::paddle::operators::UnaryLogicalOpKernel<                           \
          ::paddle::platform::dev##DeviceContext, functor<int>>,           \
      ::paddle::operators::UnaryLogicalOpKernel<                           \
          ::paddle::platform::dev##DeviceContext, functor<int64_t>>,       \
      ::paddle::operators::UnaryLogicalOpKernel<                           \
          ::paddle::platform::dev##DeviceContext, functor<float>>,         \
      ::paddle::operators::UnaryLogicalOpKernel<                           \
          ::paddle::platform::dev##DeviceContext, functor<double>>);
