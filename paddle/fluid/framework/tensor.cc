/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/tensor.h"
#include "paddle/pten/api/lib/utils/storage.h"

DECLARE_bool(use_stream_safe_cuda_allocator);

namespace paddle {
namespace framework {

Tensor Tensor::Slice(int64_t begin_idx, int64_t end_idx) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(begin_idx, 0,
                    paddle::platform::errors::OutOfRange(
                        "The start row index must be greater than 0."
                        "But received the start index is d%.",
                        begin_idx));
  PADDLE_ENFORCE_LE(end_idx, meta_.dims[0],
                    paddle::platform::errors::OutOfRange(
                        "The end row index is out of bound."));
  PADDLE_ENFORCE_LT(
      begin_idx, end_idx,
      paddle::platform::errors::InvalidArgument(
          "The start row index must be less than the end row index."
          "But received the start index = %d, the end index = %d.",
          begin_idx, end_idx));

  if (meta_.dims[0] == 1) {
    return *this;
  } else {
    size_t base = numel() / meta_.dims[0];
    Tensor dst;
    dst.storage_ = pten::make_intrusive<paddle::experimental::SharedStorage>(
        storage_->data_shared());
    dst.meta_.layout = meta_.layout;
    dst.meta_.dtype = meta_.dtype;
    DDim dst_dims = meta_.dims;
    dst_dims[0] = end_idx - begin_idx;
    dst.Resize(dst_dims);
    dst.meta_.offset = meta_.offset + begin_idx * base * SizeOf(dtype());
    return dst;
  }
}

std::vector<Tensor> Tensor::Split(int64_t split_size, int64_t axis) const {
  check_memory_size();

  PADDLE_ENFORCE_GE(meta_.dims.size(), 0,
                    paddle::platform::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));

  PADDLE_ENFORCE_GE(
      split_size, 0,
      paddle::platform::errors::OutOfRange(
          "split expects split_size be non-negative, but got split_size is %d",
          split_size));

  int64_t numel_size = meta_.dims[axis];

  int64_t num_splits = 1;
  if (split_size != 0) {
    num_splits =
        std::max<int64_t>((numel_size + split_size - 1) / split_size, 1);
  }

  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - numel_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    int64_t length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = Slice(i * split_size, i * split_size + length);
  }
  return splits;
}

std::vector<Tensor> Tensor::Chunk(int64_t chunks, int64_t axis) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(meta_.dims.size(), 0,
                    paddle::platform::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));
  PADDLE_ENFORCE_GE(
      chunks, 0,
      paddle::platform::errors::OutOfRange(
          "chunks expects to be greater than 0, but got chunks is %d", chunks));

  int64_t numel_size = meta_.dims[axis];
  int64_t split_size = (numel_size + chunks - 1) / chunks;
  return Split(split_size, axis);
}

Tensor& Tensor::ShareDataWith(const Tensor& src) {
  src.check_memory_size();
  // Preserve LoD
  auto lod = meta_.lod;
  *this = src;
  meta_.lod = lod;
  return *this;
}
Tensor& Tensor::ShareInplaceVersionCounterWith(const Tensor& src) {
  PADDLE_ENFORCE_NOT_NULL(
      inplace_version_counter_,
      platform::errors::PreconditionNotMet(
          "Tensor does not hold inplace_version_counter_."));

  inplace_version_counter_ = src.inplace_version_counter_;
  return *this;
}

}  // namespace framework
}  // namespace paddle
