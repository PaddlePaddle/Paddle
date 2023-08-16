// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

namespace phi {
namespace distributed {

void* DistTensor::AllocateFrom(Allocator* allocator,
                               DataType dtype,
                               size_t requested_size,
                               bool fake_alloc) {
  return value_->AllocateFrom(allocator, dtype, requested_size, fake_alloc);
}

const Place& DistTensor::place() const {
  PADDLE_ENFORCE_EQ(
      value_ && value_->holder_,
      true,
      phi::errors::PreconditionNotMet(
          "Tensor not initialized yet when DistTensor::place() is called."));
  return value_->holder_->place();
}

int64_t DistTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

void DistTensor::set_meta(DenseTensorMeta&& meta) {
  PADDLE_ENFORCE_EQ(meta_.valid(),
                    false,
                    phi::errors::InvalidArgument(
                        "Only when the original attribute of Tensor is "
                        "incomplete, can it be reset."));
  meta_ = std::move(meta);
}

void DistTensor::set_meta(const DenseTensorMeta& meta) {
  PADDLE_ENFORCE_EQ(
      meta.valid(),
      true,
      phi::errors::InvalidArgument(
          "Input meta is invalid, please check the meta attribute."));
  meta_ = meta;
}

}  // namespace distributed
}  // namespace phi
