/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

template <typename Context, typename T, typename IndexT>
class SegmentPoolFunctor {
 public:
  /* mean pool has summed_ids output */
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& segments,
                  DenseTensor* output,
                  DenseTensor* summed_ids = nullptr,
                  const std::string pooltype = "SUM");
};

template <typename Context, typename T, typename IndexT>
class SegmentPoolGradFunctor {
 public:
  /* mean pool has summed_ids output */
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& out_grad,
                  const DenseTensor& segments,
                  DenseTensor* in_grad,
                  paddle::optional<const DenseTensor&> summed_ids,
                  const std::string pooltype = "SUM");
};

}  // namespace funcs
}  // namespace phi
