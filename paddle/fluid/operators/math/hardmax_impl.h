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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
void HardmaxFunctor<DeviceContext, T>::operator()(const DeviceContext& context,
                                                  const framework::Tensor* X,
                                                  framework::Tensor* Y) {
  auto logits = EigenMatrix<T>::From(*X);
  auto hardmax = EigenMatrix<T>::From(*Y);
  auto shifted_logits = logits;

  for (int i = 0; logits.rows(); ++i) {
    int maxValue = logits.getRows(i).maximum();
    for (int j = 0; logits.cols(); ++j) {
      if (hardmax(i, j) == maxValue) {
        shifted_logits(i, j) = 1;
      } else {
        shifted_logits(i, j) = 0;
      }
    }
  }
  hardmax.device(*context.eigen_device()) = shifted_logits;
}

template <typename DeviceContext, typename T>
void HardmaxGradFunctor<DeviceContext, T>::operator()(
    const DeviceContext& context, const framework::Tensor* y,
    const framework::Tensor* y_grad, framework::Tensor* x_grad) {
  auto hardmax = EigenMatrix<T>::From(*y);
  auto hardmax_grad = EigenMatrix<T>::From(*y_grad);
  auto logits_grad = EigenMatrix<T>::From(*x_grad);
  auto shifted_logits = logits;

  for (int i = 0; logits.rows(); ++i) {
    int maxValue = logits.getRows(i).maximum();
    for (int j = 0; logits.cols(); ++j) {
      if (hardmax(i, j) == maxValue) {
        shifted_logits(i, j) = 1;
      } else {
        shifted_logits(i, j) = 0;
      }
    }
  }
  logits_grad.device(*context.eigen_device()) = shifted_logits;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
