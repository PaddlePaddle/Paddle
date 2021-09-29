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

namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline size_t SizeToAxis(const int axis, const framework::DDim dims) {
  size_t size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline size_t SizeFromAxis(const int axis, const framework::DDim dims) {
  size_t size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T>
struct ValueClip {
  HOSTDEVICE T operator()(const T& x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

template <typename DeviceContext, typename T>
struct LogSoftmaxFunctor {
  void operator()(const DeviceContext& context, const framework::Tensor* X,
                  framework::Tensor* Y, const int axis) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    int axis_dim = X->dims()[axis];
    const int n = SizeToAxis(axis, X->dims());
    const int d = SizeFromAxis(axis, X->dims());
    framework::DDim dim_2d{n, d};

    auto logits = EigenMatrix<T>::From(*X, dim_2d);
    auto log_softmax = EigenMatrix<T>::From(*Y, dim_2d);

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

    log_softmax.device(*context.eigen_device()) =
        log_softmax -
        log_softmax.exp()
            .eval()
            .reshape(batch_axis_remain)
            .sum(along_axis)
            .log()
            .broadcast(one_axis);
  }
};

template <typename DeviceContext, typename T>
class LogSoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Out = context.Output<framework::Tensor>("Out");
    const int rank = X->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    if (X->numel() != 0) {
      LogSoftmaxFunctor<DeviceContext, T>()(
          context.template device_context<DeviceContext>(), X, Out, axis);
    }
  }
};

template <typename DeviceContext, typename T>
struct LogSoftmaxGradFunctor {
  void operator()(const DeviceContext& context, const framework::Tensor* Y,
                  const framework::Tensor* dY, framework::Tensor* dX,
                  const int axis) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;

    const int n = SizeToAxis(axis, Y->dims());
    const int d = SizeFromAxis(axis, Y->dims());
    framework::DDim dim_2d{n, d};

    auto y = EigenMatrix<T>::From(*Y, dim_2d);
    auto dy = EigenMatrix<T>::From(*dY, dim_2d);
    auto dx = EigenMatrix<T>::From(*dX, dim_2d);

    const int axis_dim = Y->dims()[axis];
    const int batch_size = y.dimension(kBatchDim);
    const int num_classes = y.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);

    dx.device(*context.eigen_device()) =
        dy -
        (y.exp()) * (dy.reshape(batch_axis_remain)
                         .sum(along_class)
                         .broadcast(one_axis));
  }
};

template <typename DeviceContext, typename T>
class LogSoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Out = context.Input<framework::Tensor>("Out");
    auto* dOut =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<framework::Tensor>(framework::GradVarName("X"));
    const int rank = Out->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());

    if (Out->numel() != 0) {
      LogSoftmaxGradFunctor<DeviceContext, T>()(
          context.template device_context<DeviceContext>(), Out, dOut, dX,
          axis);
    }
  }
};

}  // namespace operators
}  // namespace paddle
