// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ops/item.h>

#include "paddle/phi/api/include/api.h"

namespace at {

inline bool equal(const at::Tensor& self, const at::Tensor& other) {
  PD_CHECK(self.device() == other.device(),
           "Cannot compare two tensors on "
           "different devices. Got: ",
           self.device(),
           " and ",
           other.device());
  if (self.sizes() != other.sizes()) {
    return false;
  }

  auto result = paddle::experimental::equal_all(self._PD_GetInner(),
                                                other._PD_GetInner());
  return at::Tensor(std::move(result)).item<bool>();
}

}  // namespace at

namespace at {

inline bool Tensor::equal(const at::Tensor& other) const {
  return at::equal(*this, other);
}

}  // namespace at
