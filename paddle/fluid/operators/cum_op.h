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

#include <array>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename Functor>
class CumKernel : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;

  void Compute(const framework::ExecutionContext& context) const override {
    auto& X = detail::Ref(context.Input<framework::Tensor>("X"),
                          "Cannot get input tensor X, variable name = %s",
                          context.InputName("X"));

    auto& Out = detail::Ref(context.Output<framework::Tensor>("Out"),
                            "Cannot get output tensor Out, variable name = %s",
                            context.OutputName("Out"));
    int axis = context.Attr<int>("axis");
    bool exclusive = context.Attr<bool>("exclusive");
    bool reverse = context.Attr<bool>("reverse");
    auto x_dims = X.dims();
    if (axis == -1) {
      axis = x_dims.size() - 1;
    }
    PADDLE_ENFORCE_LT(
        axis, x_dims.size(),
        "axis should be less than the dimensiotn of the input tensor");
    Out.mutable_data<T>(context.GetPlace());

    int pre = 1;
    int post = 1;
    int mid = x_dims[axis];
    for (int i = 0; i < axis; ++i) {
      pre *= x_dims[i];
    }
    for (int i = axis + 1; i < x_dims.size(); ++i) {
      post *= x_dims[i];
    }

    auto x = framework::EigenVector<T>::Flatten(X);
    auto out = framework::EigenVector<T>::Flatten(Out);
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();

    using IndexT = Eigen::DenseIndex;
    if (pre == 1) {
      if (post == 1) {
        ComputeImp(*place, Eigen::DSizes<IndexT, 1>(mid), x, out,
                   /* axis= */ 0, reverse, exclusive);
      } else {
        ComputeImp(*place, Eigen::DSizes<IndexT, 2>(mid, post), x, out,
                   /* axis= */ 0, reverse, exclusive);
      }
    } else {
      if (post == 1) {
        ComputeImp(*place, Eigen::DSizes<IndexT, 2>(pre, mid), x, out,
                   /* axis= */ 1, reverse, exclusive);
      } else {
        ComputeImp(*place, Eigen::DSizes<IndexT, 3>(pre, mid, post), x, out,
                   /* axis= */ 1, reverse, exclusive);
      }
    }
  }

 private:
  template <typename Device, typename Dim, typename X, typename Out>
  void ComputeImp(Device d, const Dim& dims, X x, Out out, int axis,
                  bool reverse, bool exclusive) const {
    if (!reverse) {
      out.reshape(dims).device(d) = Functor()(x.reshape(dims), axis, exclusive);
    } else {
      std::array<bool, Dim::count> rev;
      rev.fill(false);
      rev[axis] = reverse;
      out.reshape(dims).device(d) =
          Functor()(x.reshape(dims).reverse(rev), axis, exclusive).reverse(rev);
    }
  }
};

template <typename T>
struct CumsumFunctor {
  using ELEMENT_TYPE = T;
  template <typename X>
  const typename X::TensorScanSumOp operator()(X x, int axis,
                                               bool exclusive) const {
    return x.cumsum(axis, exclusive);
  }
};

}  // namespace operators
}  // namespace paddle
