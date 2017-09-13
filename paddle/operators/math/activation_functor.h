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
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

template <typename Place, typename T>
struct Sigmoid {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& X, framework::Tensor* Y) {
    auto x = framework::EigenVector<T>::Flatten(X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto* place = device_context.template get_eigen_device<Place>();
    y.device(*place) = 1. / (1. + (-x).exp());
  }
};

template <typename Place, typename T>
struct SigmoidGrad {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& X, const framework::Tensor& Y,
                  const framework::Tensor& dY, framework::Tensor* dX) {
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto y = framework::EigenVector<T>::Flatten(Y);
    auto dy = framework::EigenVector<T>::Flatten(dY);
    auto* place = device_context.template get_eigen_device<Place>();
    dx.device(*place) = dy * y * (1. - y);
  }
};

template <typename Place, typename T>
struct Exp {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& input, framework::Tensor* output) {
    auto x = framework::EigenVector<T>::Flatten(input);
    auto y = framework::EigenVector<T>::Flatten(*output);
    auto* place = device_context.template get_eigen_device<Place>();
    y.device(*place) = x.exp();
  }
};

template <typename Place, typename T>
struct ExpGrad {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& X, const framework::Tensor& Y,
                  const framework::Tensor& dY, framework::Tensor* dX) {
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto dy = framework::EigenVector<T>::Flatten(dY);
    auto* place = device_context.template get_eigen_device<Place>();
    dx.device(*place) = dy.exp();
  }
};

template <typename Place, typename T>
struct Relu {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& input, framework::Tensor* output) {
    auto x = framework::EigenVector<T>::Flatten(input);
    auto y = framework::EigenVector<T>::Flatten(*output);
    auto* place = device_context.template get_eigen_device<Place>();
    y.device(*place) = x.cwiseMax(static_cast<T>(0));
  }
};

template <typename Place, typename T>
struct ReluGrad {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& X, const framework::Tensor& Y,
                  const framework::Tensor& dY, framework::Tensor* dX) {
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto dy = framework::EigenVector<T>::Flatten(dY);
    auto x = framework::EigenVector<T>::Flatten(X);
    auto* place = device_context.template get_eigen_device<Place>();
    dx.device(*place) = dy * (x > static_cast<T>(0)).template cast<T>();
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
