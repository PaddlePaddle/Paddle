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

inline at::Tensor detach(const at::Tensor& self) {
  // Create a new Tensor that shares data but has no autograd history
  auto inner = self._PD_GetInner();
  PaddleTensor detached_tensor(inner.impl());
  detached_tensor.set_name(inner.name());
  detached_tensor.set_autograd_meta(nullptr);
  return Tensor(detached_tensor);
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::detach() const { return at::detach(*this); }

inline at::Tensor& Tensor::detach_() const {
  // In-place version: clear autograd meta of current tensor
  PaddleTensor& inner = const_cast<PaddleTensor&>(tensor_);
  inner.set_autograd_meta(nullptr);
  return const_cast<at::Tensor&>(*this);
}

}  // namespace at
