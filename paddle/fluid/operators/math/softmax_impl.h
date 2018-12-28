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

#include "paddle/fluid/operators/math/blas.h"
namespace paddle {
namespace operators {
namespace math {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct ValueClip {
  HOSTDEVICE T operator()(const T& x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

template <typename DeviceContext, typename T, bool is_test, typename Enable>
void SoftmaxFunctor<DeviceContext, T, is_test, Enable>::operator()(
    const DeviceContext& context, const framework::Tensor* X,
    framework::Tensor* Y) {
  auto logits = EigenMatrix<T>::From(*X);
  auto softmax = EigenMatrix<T>::From(*Y);

  const int kBatchDim = 0;
  const int kClassDim = 1;

  const int batch_size = logits.dimension(kBatchDim);
  const int num_classes = logits.dimension(kClassDim);

  Eigen::DSizes<int, 1> along_class(kClassDim);
  Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
  Eigen::DSizes<int, 2> one_by_class(1, num_classes);

  auto shifted_logits = (logits -
                         logits.maximum(along_class)
                             .eval()
                             .reshape(batch_by_one)
                             .broadcast(one_by_class))
                            .unaryExpr(ValueClip<T>());

  softmax.device(*context.eigen_device()) = shifted_logits.exp();
  softmax.device(*context.eigen_device()) = (softmax *
                                             softmax.sum(along_class)
                                                 .inverse()
                                                 .eval()
                                                 .reshape(batch_by_one)
                                                 .broadcast(one_by_class));
}

template <class DeviceContext>
using enable_if_CPU = typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type;

template <typename DeviceContext>
class SoftmaxFunctor<DeviceContext, float, true, enable_if_CPU<DeviceContext>> {
  void operator()(const DeviceContext& context, const framework::Tensor* X,
                  framework::Tensor* Y) {
    auto in_dims = X->dims();
    const float* in_data = X->data<float>();
    float* out_data = Y->data<float>();
    const int kBatchDim = 0;
    const int kClassDim = 1;
    // 2D data. Batch x C
    const int batch_size = in_dims[kBatchDim];
    const int num_classes = in_dims[kClassDim];
    std::vector<float> entities(batch_size);
    auto blas = math::GetBlas<DeviceContext, float>(context);
    for (int n = 0; n < batch_size; ++n) {
      entities[n] = in_data[n * num_classes];
      for (int c = 1; c < num_classes; ++c) {
        entities[n] = in_data[n * num_classes + c] > entities[n]
                          ? in_data[n * num_classes + c]
                          : entities[n];
      }
      for (int c = 0; c < num_classes; ++c) {
        out_data[n * num_classes + c] =
            in_data[n * num_classes + c] - entities[n];
      }
    }

    blas.VEXP(num_classes * batch_size, out_data, out_data);
    for (int n = 0; n < batch_size; ++n) {
      auto sum = blas.ASUM(num_classes, &out_data[n * num_classes], 1);
      blas.SCAL(num_classes, 1.0f / sum, &out_data[n * num_classes]);
    }
  }
};

template <typename DeviceContext, typename T>
void SoftmaxGradFunctor<DeviceContext, T>::operator()(
    const DeviceContext& context, const framework::Tensor* y,
    const framework::Tensor* y_grad, framework::Tensor* x_grad) {
  auto softmax = EigenMatrix<T>::From(*y);
  auto softmax_grad = EigenMatrix<T>::From(*y_grad);
  auto logits_grad = EigenMatrix<T>::From(*x_grad);

  const int kBatchDim = 0;
  const int kClassDim = 1;

  const int batch_size = softmax.dimension(kBatchDim);
  const int num_classes = softmax.dimension(kClassDim);

  Eigen::DSizes<int, 1> along_class(kClassDim);
  Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
  Eigen::DSizes<int, 2> one_by_class(1, num_classes);

  auto dot = (softmax * softmax_grad)
                 .sum(along_class)
                 .eval()
                 .reshape(batch_by_one)
                 .broadcast(one_by_class);
  logits_grad.device(*context.eigen_device()) = (softmax_grad - dot) * softmax;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
