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

inline at::Tensor t(const Tensor& self) {
  return self.transpose(0, self.dim() < 2 ? 0 : 1);
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::t() const { return at::t(*this); }

inline at::Tensor& Tensor::t_() const {
  return transpose_(0, dim() < 2 ? 0 : 1);
}

}  // namespace at
