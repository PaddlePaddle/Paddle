/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/elementwise_op_function.h"
#include "paddle/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
struct LessThanFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a, const T& b) const { return a < b; }
};

template <typename T>
struct LessEqualFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a, const T& b) const { return a <= b; }
};

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

template <typename DeviceContext, typename Functor>
class CompareOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    z->mutable_data<T>(context.GetPlace());
    int axis = context.Attr<int>("axis");
    ElementwiseComputeEx<Functor, DeviceContext, T, bool>(context, x, y, axis,
                                                          z);
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_LOGICAL_KERNEL(op_type, dev, functor)                    \
  REGISTER_OP_##dev##_KERNEL(                                             \
      op_type, ::paddle::operators::CompareOpKernel<                      \
                   ::paddle::platform::dev##DeviceContext, functor<int>>, \
      ::paddle::operators::CompareOpKernel<                               \
          ::paddle::platform::dev##DeviceContext, functor<int64_t>>,      \
      ::paddle::operators::CompareOpKernel<                               \
          ::paddle::platform::dev##DeviceContext, functor<float>>,        \
      ::paddle::operators::CompareOpKernel<                               \
          ::paddle::platform::dev##DeviceContext, functor<double>>);
