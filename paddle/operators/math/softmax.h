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
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

template <typename Place, typename T>
class SoftmaxFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor* X, framework::Tensor* Y);
};

template <typename Place, typename T>
class SoftmaxGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor* y, const framework::Tensor* y_grad,
                  framework::Tensor* x_grad);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
