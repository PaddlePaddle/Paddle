/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/softmax_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class LogSoftmaxEigen {
 public:
  void operator()(const DeviceContext& context, const int axis_dim,
                  const framework::Tensor* X, framework::Tensor* Y) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    auto logits = EigenMatrix<T>::From(*X);
    auto log_softmax = EigenMatrix<T>::From(*Y);

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_axis(kAxisDim);
    Eigen::DSizes<int, 2> batch_classes(batch_size, num_classes);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
    Eigen::DSizes<int, 3> batch_one_remain(batch_size, 1, num_remain);
    Eigen::DSizes<int, 3> one_axis_one(1, axis_dim, 1);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);

    // For numerical stability, logits should be shifted by maximum number along
    // axis, calculate shifted_logits into log_softmax tensor for memory reuse.
    if (num_remain == 1) {
      // axis == -1, axis and class in same dimension, calculate along
      // class dimension directly for higher performance
      log_softmax.device(*context.eigen_device()) =
          (logits -
           logits.maximum(along_axis)
               .eval()
               .reshape(batch_by_one)
               .broadcast(one_by_class))
              .unaryExpr(ValueClip<T>());
    } else {
      // axis != -1, class dimension split into (axis, remain), max and sum
      // should be calculated along axis dimension
      log_softmax.device(*context.eigen_device()) =
          (logits.reshape(batch_axis_remain) -
           logits.reshape(batch_axis_remain)
               .maximum(along_axis)
               .eval()
               .reshape(batch_one_remain)
               .broadcast(one_axis_one)
               .reshape(batch_classes))
              .unaryExpr(ValueClip<T>());
    }

    softmax.device(*context.eigen_device()) = softmax.exp();
    softmax.device(*context.eigen_device()) =
        (softmax *
         softmax.reshape(batch_axis_remain)
             .sum(along_axis)
             .inverse()
             .eval()
             .broadcast(one_axis));
  }
};

template <typename DeviceContext, typename T>
class LogSoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    auto x_dims = X->dims();
    const int rank = x_dims.size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = x_dims[axis];

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, X->dims());
    const int d = SizeFromAxis(axis, X->dims());
    Tensor X_2d, Out_2d;
    X_2d.ShareDataWith(*X).Resize({n, d});
    Out_2d.ShareDataWith(*Out).Resize({n, d});

    math::SoftmaxFunctor<DeviceContext, T, false>()(
        context.template device_context<DeviceContext>(), axis_dim, &X_2d,
        &Out_2d);
  }
};

}  // namespace operators
}  // namespace paddle
