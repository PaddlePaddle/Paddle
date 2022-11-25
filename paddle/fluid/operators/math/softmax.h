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
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

<<<<<<< HEAD
template <typename DeviceContext, typename T, typename Enable = void>
=======
template <typename DeviceContext,
          typename T,
          bool is_test,
          typename Enable = void>
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
class SoftmaxFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
<<<<<<< HEAD
                  const phi::DenseTensor* X,
                  phi::DenseTensor* Y);
=======
                  const framework::Tensor* X,
                  framework::Tensor* Y);
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

template <typename DeviceContext, typename T, typename Enable = void>
class SoftmaxGradFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const int axis_dim,
<<<<<<< HEAD
                  const phi::DenseTensor* y,
                  const phi::DenseTensor* y_grad,
                  phi::DenseTensor* x_grad);
=======
                  const framework::Tensor* y,
                  const framework::Tensor* y_grad,
                  framework::Tensor* x_grad);
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T, typename DeviceContext>
class SoftmaxCUDNNFunctor {
 public:
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor* X,
                  phi::DenseTensor* Y);
=======
                  const framework::Tensor* X,
                  framework::Tensor* Y);
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

template <typename T, typename DeviceContext>
class SoftmaxGradCUDNNFunctor {
 public:
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor* Y,
                  const phi::DenseTensor* y_grad,
                  phi::DenseTensor* x_grad);
=======
                  const framework::Tensor* Y,
                  const framework::Tensor* y_grad,
                  framework::Tensor* x_grad);
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
