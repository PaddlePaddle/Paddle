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
#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
class SequencePoolFunctor {
 public:
  /* max pool has index output */
  void operator()(const DeviceContext& context, const std::string pooltype,
                  const framework::LoDTensor& input, framework::Tensor* output,
                  framework::Tensor* index = nullptr);
};

template <typename DeviceContext, typename T>
class SequencePoolGradFunctor {
 public:
  void operator()(const DeviceContext& context, const std::string pooltype,
                  const framework::Tensor& out_grad,
                  framework::LoDTensor* in_grad,
                  /* max pool has index */
                  const framework::Tensor* index = nullptr);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
