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
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct ValueClip {
  HOSTDEVICE T operator()(const T& x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

template <typename DeviceContext, typename T>
class SoftmaxEigen {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* X,
                  phi::DenseTensor* Y) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    auto logits = EigenMatrix<T>::From(*X);
    auto softmax = EigenMatrix<T>::From(*Y);

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
    // axis, calculate shifted_logits into softmax tensor for memory reuse.
    if (num_remain == 1) {
      // axis == -1, axis and class in same dimension, calculate along
      // class dimension directly for higher performance
      softmax.device(*context.eigen_device()) =
          (logits - logits.maximum(along_axis)
                        .eval()
                        .reshape(batch_by_one)
                        .broadcast(one_by_class))
              .unaryExpr(ValueClip<T>());
    } else {
      // axis != -1, class dimension split into (axis, remain), max and sum
      // should be calculated along axis dimension
      softmax.device(*context.eigen_device()) =
          (logits.reshape(batch_axis_remain) - logits.reshape(batch_axis_remain)
                                                   .maximum(along_axis)
                                                   .eval()
                                                   .reshape(batch_one_remain)
                                                   .broadcast(one_axis_one)
                                                   .reshape(batch_classes))
              .unaryExpr(ValueClip<T>());
    }

    softmax.device(*context.eigen_device()) = softmax.exp();
    softmax.device(*context.eigen_device()) =
        (softmax * softmax.reshape(batch_axis_remain)
                       .sum(along_axis)
                       .inverse()
                       .eval()
                       .broadcast(one_axis));
  }
};

template <typename DeviceContext>
class SoftmaxEigen<DeviceContext, platform::float16> {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* X,
                  phi::DenseTensor* Y) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    auto logits = EigenMatrix<platform::float16>::From(*X);
    auto softmax = EigenMatrix<platform::float16>::From(*Y);

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
    // axis, calculate shifted_logits into softmax tensor for memory reuse.
    if (num_remain == 1) {
      // axis == -1, axis and class in same dimension, calculate along
      // class dimension directly for higher performance
      softmax.device(*context.eigen_device()) =
          (logits - logits.maximum(along_axis)
                        .reshape(batch_by_one)
                        .broadcast(one_by_class))
              .unaryExpr(ValueClip<platform::float16>());
    } else {
      // axis != -1, class dimension split into (axis, remain), max and sum
      // should be calculated along axis dimension
      softmax.device(*context.eigen_device()) =
          (logits.reshape(batch_axis_remain) - logits.reshape(batch_axis_remain)
                                                   .maximum(along_axis)
                                                   .reshape(batch_one_remain)
                                                   .broadcast(one_axis_one)
                                                   .reshape(batch_classes))
              .unaryExpr(ValueClip<platform::float16>());
    }

    softmax.device(*context.eigen_device()) = softmax.exp();
    softmax.device(*context.eigen_device()) =
        (softmax * softmax.reshape(batch_axis_remain)
                       .sum(along_axis)
                       .inverse()
                       .broadcast(one_axis));
  }
};

template <typename DeviceContext>
class SoftmaxEigen<DeviceContext, platform::bfloat16> {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* X,
                  phi::DenseTensor* Y) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    auto logits = EigenMatrix<platform::bfloat16>::From(*X);
    auto softmax = EigenMatrix<platform::bfloat16>::From(*Y);

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
    // axis, calculate shifted_logits into softmax tensor for memory reuse.
    if (num_remain == 1) {
      // axis == -1, axis and class in same dimension, calculate along
      // class dimension directly for higher performance
      softmax.device(*context.eigen_device()) =
          (logits - logits.maximum(along_axis)
                        .reshape(batch_by_one)
                        .broadcast(one_by_class))
              .unaryExpr(ValueClip<platform::bfloat16>());
    } else {
      // axis != -1, class dimension split into (axis, remain), max and sum
      // should be calculated along axis dimension
      softmax.device(*context.eigen_device()) =
          (logits.reshape(batch_axis_remain) - logits.reshape(batch_axis_remain)
                                                   .maximum(along_axis)
                                                   .reshape(batch_one_remain)
                                                   .broadcast(one_axis_one)
                                                   .reshape(batch_classes))
              .unaryExpr(ValueClip<platform::bfloat16>());
    }

    softmax.device(*context.eigen_device()) = softmax.exp();
    softmax.device(*context.eigen_device()) =
        (softmax * softmax.reshape(batch_axis_remain)
                       .sum(along_axis)
                       .inverse()
                       .broadcast(one_axis));
  }
};

template <typename DeviceContext, typename T, typename Enable>
void SoftmaxFunctor<DeviceContext, T, Enable>::operator()(
    const DeviceContext& context,
    const int axis_dim,
    const phi::DenseTensor* X,
    phi::DenseTensor* Y) {
  SoftmaxEigen<DeviceContext, T>()(context, axis_dim, X, Y);
}

template <class DeviceContext>
using enable_if_CPU = typename std::enable_if<
    std::is_same<DeviceContext, phi::CPUContext>::value>::type;

template <typename DeviceContext, typename T>
class SoftmaxFunctor<DeviceContext, T, enable_if_CPU<DeviceContext>> {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* X,
                  phi::DenseTensor* Y) {
    const auto& in_dims = X->dims();
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;

    const int num_classes = in_dims[kClassDim];
    const int batch_size = in_dims[kBatchDim];
    const int num_remain = num_classes / axis_dim;

    if (num_remain == 1 && platform::MayIUse(platform::avx)) {
      const T* in_data = X->data<T>();
      T* out_data = Y->data<T>();
      for (int bs = 0; bs < batch_size; ++bs) {
        T max_val = *std::max_element(in_data, in_data + num_classes);
        max_val *= static_cast<T>(-1);
        vec_add_bias<T, platform::avx>(num_classes, max_val, in_data, out_data);
        vec_clip<T, platform::avx>(
            num_classes, static_cast<T>(-64), out_data, out_data);
        vec_exp<T>(num_classes, out_data, out_data);

        T sum = 0;
        vec_sum<T, platform::avx>(num_classes, out_data, &sum);
        sum = static_cast<T>(1) / sum;
        vec_scal<T, platform::avx>(num_classes, sum, out_data, out_data);

        in_data += num_classes;
        out_data += num_classes;
      }
    } else {
      SoftmaxEigen<DeviceContext, T>()(context, axis_dim, X, Y);
    }
  }
};

template <typename DeviceContext, typename T>
class SoftmaxGradEigen {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* y,
                  const phi::DenseTensor* y_grad,
                  phi::DenseTensor* x_grad) {
    auto softmax = EigenMatrix<T>::From(*y);
    auto softmax_grad = EigenMatrix<T>::From(*y_grad);
    auto logits_grad = EigenMatrix<T>::From(*x_grad);

    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;

    const int batch_size = softmax.dimension(kBatchDim);
    const int num_classes = softmax.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);

    auto dot = (softmax * softmax_grad)
                   .reshape(batch_axis_remain)
                   .sum(along_class)
                   .eval()
                   .broadcast(one_axis);
    logits_grad.device(*context.eigen_device()) =
        (softmax_grad - dot) * softmax;
  }
};

template <typename DeviceContext>
class SoftmaxGradEigen<DeviceContext, platform::float16> {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* y,
                  const phi::DenseTensor* y_grad,
                  phi::DenseTensor* x_grad) {
    auto softmax = EigenMatrix<platform::float16>::From(*y);
    auto softmax_grad = EigenMatrix<platform::float16>::From(*y_grad);
    auto logits_grad = EigenMatrix<platform::float16>::From(*x_grad);

    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;

    const int batch_size = softmax.dimension(kBatchDim);
    const int num_classes = softmax.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);

    auto dot = (softmax * softmax_grad)
                   .reshape(batch_axis_remain)
                   .sum(along_class)
                   .broadcast(one_axis);
    logits_grad.device(*context.eigen_device()) =
        (softmax_grad - dot) * softmax;
  }
};

template <typename DeviceContext>
class SoftmaxGradEigen<DeviceContext, platform::bfloat16> {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* y,
                  const phi::DenseTensor* y_grad,
                  phi::DenseTensor* x_grad) {
    auto softmax = EigenMatrix<platform::bfloat16>::From(*y);
    auto softmax_grad = EigenMatrix<platform::bfloat16>::From(*y_grad);
    auto logits_grad = EigenMatrix<platform::bfloat16>::From(*x_grad);

    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;

    const int batch_size = softmax.dimension(kBatchDim);
    const int num_classes = softmax.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);

    auto dot = (softmax * softmax_grad)
                   .reshape(batch_axis_remain)
                   .sum(along_class)
                   .broadcast(one_axis);
    logits_grad.device(*context.eigen_device()) =
        (softmax_grad - dot) * softmax;
  }
};

template <typename DeviceContext, typename T, typename Enable>
void SoftmaxGradFunctor<DeviceContext, T, Enable>::operator()(
    const DeviceContext& context,
    const int axis_dim,
    const phi::DenseTensor* y,
    const phi::DenseTensor* y_grad,
    phi::DenseTensor* x_grad) {
  SoftmaxGradEigen<DeviceContext, T>()(context, axis_dim, y, y_grad, x_grad);
}

template <typename DeviceContext, typename T>
class SoftmaxGradFunctor<DeviceContext, T, enable_if_CPU<DeviceContext>> {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
                  const phi::DenseTensor* y,
                  const phi::DenseTensor* y_grad,
                  phi::DenseTensor* x_grad) {
    const auto& out_dims = y->dims();
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    const int num_classes = out_dims[kClassDim];
    const int batch_size = out_dims[kBatchDim];
    const int num_remain = num_classes / axis_dim;

    if (num_remain == 1 && platform::MayIUse(platform::avx)) {
      const T* out_data = y->data<T>();
      const T* out_grad = y_grad->data<T>();
      T* in_grad = x_grad->data<T>();
      for (int bs = 0; bs < batch_size; ++bs) {
        T scalar;
        vec_mul_reduce<T, platform::avx>(
            num_classes, out_grad, out_data, &scalar);
        scalar *= static_cast<T>(-1);
        vec_add_bias<T, platform::avx>(num_classes, scalar, out_grad, in_grad);
        vec_mul<T, platform::avx>(num_classes, out_data, in_grad, in_grad);
        out_data += num_classes;
        out_grad += num_classes;
        in_grad += num_classes;
      }
    } else {
      SoftmaxGradEigen<DeviceContext, T>()(
          context, axis_dim, y, y_grad, x_grad);
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
