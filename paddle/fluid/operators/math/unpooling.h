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
template <typename DeviceContext, typename T>
class Unpool2dMaxFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& indices, framework::Tensor* output);
};
template <typename DeviceContext, class T>
class Unpool2dMaxGradFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& indices,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  framework::Tensor* input_grad);
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
