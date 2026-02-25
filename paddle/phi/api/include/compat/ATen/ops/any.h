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
#include <c10/core/Scalar.h>
#include <c10/util/OptionalArrayRef.h>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"

namespace paddle {
namespace experimental {

// any - returns true if any element is non-zero
// Implementation using phi::AnyKernel
inline Tensor any(const Tensor& self) {
  auto& dense = self._PD_GetInner();
  phi::DenseTensor out;
  std::vector<int64_t> dims;
  phi::AnyKernel<bool, phi::CPUContext>(
      phi::CPUContext(), dense, dims, false, &out);
  return Tensor(out);
}

inline Tensor any(const Tensor& self,
                  const std::vector<int64_t>& dims,
                  bool keepdim = false) {
  auto& dense = self._PD_GetInner();
  phi::DenseTensor out;
  phi::AnyKernel<bool, phi::CPUContext>(
      phi::CPUContext(), dense, dims, keepdim, &out);
  return Tensor(out);
}

inline Tensor any(const phi::IntArray& dims, bool keepdim, const Tensor& self) {
  auto& dense = self._PD_GetInner();
  phi::DenseTensor out;
  std::vector<int64_t> dims_vec = dims.GetData();
  phi::AnyKernel<bool, phi::CPUContext>(
      phi::CPUContext(), dense, dims_vec, keepdim, &out);
  return Tensor(out);
}

}  // namespace experimental
}  // namespace paddle

namespace at {

// any - returns true if any element is non-zero (free functions)
inline Tensor any(const Tensor& self, int64_t dim, bool keepdim = false) {
  return paddle::experimental::any(self, {dim}, keepdim);
}

inline Tensor any(const Tensor& self,
                  at::OptionalIntArrayRef dim,
                  bool keepdim = false) {
  std::vector<int64_t> dims_vec;
  if (dim.has_value() && dim.value().size() > 0) {
    dims_vec.assign(dim.value().begin(), dim.value().end());
  }
  return paddle::experimental::any(self, dims_vec, keepdim);
}

inline Tensor any(const Tensor& self) {
  return paddle::experimental::any(self);
}

}  // namespace at

namespace at {

// Member function implementations for Tensor class
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
  return paddle::experimental::any(*this, {dim}, keepdim);
}

inline Tensor Tensor::any(at::OptionalIntArrayRef dim, bool keepdim) const {
  std::vector<int64_t> dims_vec;
  if (dim.has_value() && dim.value().size() > 0) {
    dims_vec.assign(dim.value().begin(), dim.value().end());
  }
  return paddle::experimental::any(*this, dims_vec, keepdim);
}

inline Tensor Tensor::any() const { return paddle::experimental::any(*this); }

}  // namespace at
