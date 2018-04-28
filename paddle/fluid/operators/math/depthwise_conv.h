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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * \brief Compute the depthwise convolution which include
 * forward process and backpropagation process
 */
template <typename DeviceContext, typename T>
class DepthwiseConvFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, framework::Tensor* output);
};

template <typename DeviceContext, typename T>
class DepthwiseConvInputGradFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& filter,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  framework::Tensor* input_grad);
};

template <typename DeviceContext, typename T>
class DepthwiseConvFilterGradFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  framework::Tensor* filter_grad);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
