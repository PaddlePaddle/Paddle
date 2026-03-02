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

namespace at {

inline at::Tensor masked_select(const at::Tensor self, const at::Tensor& mask) {
  return Tensor(paddle::experimental::masked_select(self._PD_GetInner(),
                                                    mask._PD_GetInner()));
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::masked_select(const at::Tensor& mask) const {
  return at::masked_select(*this, mask);
}

}  // namespace at
