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
#include "paddle/phi/kernels/funcs/strided_view_utils.h"

namespace at {

// as_strided: Create a tensor view with custom size, stride, and storage_offset
inline at::Tensor Tensor::as_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    ::std::optional<int64_t> storage_offset) const {
  // Materialize the compat StorageHolderView before creating the view so
  // aliasing tensors share one StorageImpl and observe later resize_ growth.
  (void)this->storage();
  auto src_impl = tensor_.impl();
  auto* src_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(src_impl).get();
  if (!src_tensor) {
    PD_THROW("as_strided: tensor must be a DenseTensor");
  }
  // Create new meta with desired shape and strides first
  std::vector<int64_t> size_vec(size.begin(), size.end());
  std::vector<int64_t> stride_vec(stride.begin(), stride.end());

  // Create new DenseTensor with correct meta, then share data
  // We need to create a temporary DenseTensor with the right meta
  // because ShareDataWith copies the source meta which we don't want
  auto new_tensor = std::make_shared<phi::DenseTensor>();

  // First, set up the holder by sharing data (this copies src meta, we'll
  // override)
  new_tensor->ShareDataWith(*src_tensor);

  // Now create the correct meta with new shape/strides
  phi::DenseTensorMeta meta(src_tensor->dtype(),
                            common::make_ddim(size_vec),
                            common::make_ddim(stride_vec));
  // Calculate offset in bytes. `storage_offset` counts elements and is
  // relative to the source tensor, so the conversion has to be checked: an
  // unchecked multiplication wraps around and turns an out of range request
  // into a silent alias of the first element.
  int64_t offset = storage_offset.has_value() ? storage_offset.value() : 0;
  int64_t byte_offset = phi::ElementOffsetToByteOffset(
      static_cast<int64_t>(src_tensor->meta().offset),
      offset,
      static_cast<int64_t>(phi::SizeOf(src_tensor->dtype())));
  meta.offset = static_cast<size_t>(byte_offset);
  // This entry builds the view by hand instead of going through
  // phi::AsStridedKernel, so it has to run the same storage range check;
  // otherwise a view reaching past the allocation corrupts neighbouring heap
  // memory on every read and write. torch does the same in setStrided ->
  // checkInBoundsForStorage.
  phi::ValidateStridedViewStorage(
      size_vec, stride_vec, byte_offset, *src_tensor);
  new_tensor->set_meta(meta);
  PaddleTensor result;
  result.set_impl(new_tensor);
  return Tensor(result);
}

// as_strided_: Inplace version
inline const at::Tensor& Tensor::as_strided_(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    ::std::optional<int64_t> storage_offset) const {
  // Keep inplace metadata-only view rewrites attached to the same compat
  // storage as the original tensor.
  (void)this->storage();
  auto src_impl = tensor_.impl();
  auto* src_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(src_impl).get();
  if (!src_tensor) {
    PD_THROW("as_strided_: tensor must be a DenseTensor");
  }
  std::vector<int64_t> size_vec(size.begin(), size.end());
  std::vector<int64_t> stride_vec(stride.begin(), stride.end());
  // Use set_meta instead of Resize + set_strides to avoid contiguous check
  phi::DenseTensorMeta meta(src_tensor->dtype(),
                            common::make_ddim(size_vec),
                            common::make_ddim(stride_vec));
  meta.layout = src_tensor->layout();
  int64_t offset = storage_offset.has_value() ? storage_offset.value() : 0;
  int64_t byte_offset = phi::ElementOffsetToByteOffset(
      static_cast<int64_t>(src_tensor->meta().offset),
      offset,
      static_cast<int64_t>(phi::SizeOf(src_tensor->dtype())));
  meta.offset = static_cast<size_t>(byte_offset);
  phi::ValidateStridedViewStorage(
      size_vec, stride_vec, byte_offset, *src_tensor);
  src_tensor->set_meta(meta);
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
  return copy_tensor;
}

}  // namespace at
