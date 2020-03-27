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
#include <algorithm>
#include <type_traits>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
struct EqualReduceFunctor {
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
class CompareReduceOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    int axis = context.Attr<int>("axis");

    Tensor tmp;
    framework::DDim x_dims = x->dims();
    framework::DDim y_dims = y->dims();
    int max_dim = std::max(x_dims.size(), y_dims.size());
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> tmp_dims_array(max_dim);
    GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                           y_dims_array.data(), tmp_dims_array.data(), max_dim,
                           axis);
    tmp.mutable_data<bool>(framework::make_ddim(tmp_dims_array),
                           context.GetPlace());
    ElementwiseComputeEx<Functor, DeviceContext, T, bool>(context, x, y, axis,
                                                          Functor(), &tmp);

    // Reduce by 'logical and' operator
    z->mutable_data<bool>(context.GetPlace());
    auto ipt = framework::EigenVector<bool>::Flatten(tmp);
    auto out = framework::EigenScalar<bool>::From(*z);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto reduce_dim = Eigen::array<int, 1>({{0}});
    out.device(place) = ipt.all(reduce_dim);
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_REDUCE_KERNEL(op_type, dev, functor)             \
  REGISTER_OP_##dev##_KERNEL(                                             \
      op_type, ::paddle::operators::CompareReduceOpKernel<                \
                   ::paddle::platform::dev##DeviceContext, functor<int>>, \
      ::paddle::operators::CompareReduceOpKernel<                         \
          ::paddle::platform::dev##DeviceContext, functor<int64_t>>,      \
      ::paddle::operators::CompareReduceOpKernel<                         \
          ::paddle::platform::dev##DeviceContext, functor<float>>,        \
      ::paddle::operators::CompareReduceOpKernel<                         \
          ::paddle::platform::dev##DeviceContext, functor<double>>);
