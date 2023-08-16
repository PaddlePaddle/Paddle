/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

template <typename DeviceContext, typename T>
class SequencePoolFunctor {
 public:
  /* max pool has index output */
  void operator()(const DeviceContext& context,
                  const std::string pooltype,
                  T pad_value,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* output,
                  bool is_test = false,
                  phi::DenseTensor* index = nullptr);
};

template <typename DeviceContext, typename T>
class SequencePoolGradFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const std::string pooltype,
                  const phi::DenseTensor& out_grad,
                  phi::DenseTensor* in_grad,
                  /* max pool has index */
                  const phi::DenseTensor* index = nullptr);
};

}  // namespace funcs
}  // namespace phi
