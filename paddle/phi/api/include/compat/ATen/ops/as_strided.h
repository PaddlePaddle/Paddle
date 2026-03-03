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
#include <c10/util/ArrayRef.h>
#include <optional>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

namespace at {

// as_strided: Create a tensor view with custom size, stride, and storage_offset
inline at::Tensor Tensor::as_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    ::std::optional<int64_t> storage_offset) const {
  auto src_impl = tensor_.impl();
  auto* src_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(src_impl).get();
  if (!src_tensor) {
    PD_THROW("as_strided: tensor must be a DenseTensor");
  }
  auto new_tensor = std::make_shared<phi::DenseTensor>();
  new_tensor->ShareDataWith(*src_tensor);
  std::vector<int64_t> size_vec(size.begin(), size.end());
  std::vector<int64_t> stride_vec(stride.begin(), stride.end());
  new_tensor->Resize(common::make_ddim(size_vec));
  new_tensor->set_strides(common::make_ddim(stride_vec));
  int64_t offset = storage_offset.has_value() ? storage_offset.value() : 0;
  if (offset != 0) {
    auto meta = phi::DenseTensorMeta(new_tensor->meta());
    // meta.offset is in bytes; storage_offset is in elements
    meta.offset =
        static_cast<size_t>(offset) * phi::SizeOf(new_tensor->dtype());
    new_tensor->set_meta(meta);
  }
  PaddleTensor result;
  result.set_impl(new_tensor);
  return Tensor(result);
}

// as_strided_: Inplace version
inline const at::Tensor& Tensor::as_strided_(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    ::std::optional<int64_t> storage_offset) const {
  auto src_impl = tensor_.impl();
  auto* src_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(src_impl).get();
  if (!src_tensor) {
    PD_THROW("as_strided_: tensor must be a DenseTensor");
  }
  std::vector<int64_t> size_vec(size.begin(), size.end());
  std::vector<int64_t> stride_vec(stride.begin(), stride.end());
  src_tensor->Resize(common::make_ddim(size_vec));
  src_tensor->set_strides(common::make_ddim(stride_vec));
  int64_t offset = storage_offset.has_value() ? storage_offset.value() : 0;
  if (offset != 0) {
    auto meta = phi::DenseTensorMeta(src_tensor->meta());
    // meta.offset is in bytes; storage_offset is in elements
    meta.offset =
        static_cast<size_t>(offset) * phi::SizeOf(src_tensor->dtype());
    src_tensor->set_meta(meta);
  }
  return *this;
}

// as_strided_scatter: Scatter src into a strided view
// Returns a new tensor (copy of self) with the strided window filled by src.
// The original tensor is NOT modified.
inline at::Tensor Tensor::as_strided_scatter(
    const at::Tensor& src,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    ::std::optional<int64_t> storage_offset) const {
  // Clone self to an independent copy so the original tensor is left unchanged
  PaddleTensor self_copy = tensor_.copy_to(tensor_.place(), /*blocking=*/true);
  at::Tensor copy_tensor(self_copy);
  at::Tensor strided_view =
      copy_tensor.as_strided(size, stride, storage_offset);
  strided_view.copy_(src);
  return strided_view;
}

}  // namespace at
