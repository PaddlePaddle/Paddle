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
#include "paddle/common/macros.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

template <typename DeviceContext, typename T>
class MaxOutFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* output,
                  const int groups,
                  const int axis = 1);
};

template <typename DeviceContext, typename T>
class MaxOutGradFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* input_grad,
                  const phi::DenseTensor& output,
                  const phi::DenseTensor& output_grad,
                  const int groups,
                  const int axis = 1);
};
}  // namespace funcs
}  // namespace phi
