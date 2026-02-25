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

inline at::Tensor squeeze(const at::Tensor& self) {
  return paddle::experimental::squeeze(self._PD_GetInner(), {});
}

inline at::Tensor squeeze(const at::Tensor& self, int64_t dim) {
  return paddle::experimental::squeeze(self._PD_GetInner(), {dim});
}

inline at::Tensor squeeze(const at::Tensor& self, at::IntArrayRef dim) {
  return paddle::experimental::squeeze(self._PD_GetInner(),
                                       dim._PD_ToPaddleIntArray());
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::squeeze() const { return at::squeeze(*this); }

inline at::Tensor Tensor::squeeze(int64_t dim) const {
  return at::squeeze(*this, dim);
}

inline at::Tensor Tensor::squeeze(at::IntArrayRef dim) const {
  return at::squeeze(*this, dim);
}

inline at::Tensor& Tensor::squeeze_() const {
  PaddleTensor& inner = const_cast<PaddleTensor&>(tensor_);
  paddle::experimental::squeeze_(inner, {});
  return const_cast<at::Tensor&>(*this);
}

inline at::Tensor& Tensor::squeeze_(int64_t dim) const {
  PaddleTensor& inner = const_cast<PaddleTensor&>(tensor_);
  paddle::experimental::squeeze_(inner, {dim});
  return const_cast<at::Tensor&>(*this);
}

inline at::Tensor& Tensor::squeeze_(at::IntArrayRef dim) const {
  PaddleTensor& inner = const_cast<PaddleTensor&>(tensor_);
  paddle::experimental::squeeze_(inner, dim._PD_ToPaddleIntArray());
  return const_cast<at::Tensor&>(*this);
}

}  // namespace at
