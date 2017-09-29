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
#include "paddle/platform/device_context.h"
#include "paddle/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {
//////////////////////
#define FLT_MAX __FLT_MAX__

template <typename Place, typename T>
class MaxPool2dWithIndexFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

template <typename Place, typename T>
class MaxPool2dWithIndexGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

template <typename Place, typename T>
class MaxPool3dWithIndexFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

template <typename Place, typename T>
class MaxPool3dWithIndexGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
