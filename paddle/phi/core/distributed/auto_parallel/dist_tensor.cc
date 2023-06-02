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
namespace auto_parallel {

DistTensor::DistTensor(Allocator* a,
                       const DenseTensorMeta& meta,
                       const std::shared_ptr<TensorDistAttr>& dist_attr)
    : meta_(meta), dist_attr_(dist_attr) {
  // TODO(dev): value_ should only contain local tensor
  // after we have reshard.
  value_ = std::make_unique<DenseTensor>(a, meta);
}

DistTensor::DistTensor(Allocator* a,
                       DenseTensorMeta&& meta,
                       const std::shared_ptr<TensorDistAttr>& dist_attr)
    : meta_(std::move(meta)), dist_attr_(dist_attr) {
  value_ = std::make_unique<DenseTensor>(a, meta);
}

DistTensor::DistTensor(const std::shared_ptr<phi::Allocation>& holder,
                       const DenseTensorMeta& meta,
                       const std::shared_ptr<TensorDistAttr>& dist_attr)
    : meta_(meta), dist_attr_(dist_attr) {
  value_ = std::make_unique<DenseTensor>(holder, meta);
}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& dense_tensor,
                       const std::shared_ptr<TensorDistAttr>& dist_attr)
    : dist_attr_(dist_attr) {
  value_ = std::make_unique<DenseTensor>(*dense_tensor);
  set_meta(dense_tensor->meta());
}

void* DistTensor::AllocateFrom(Allocator* allocator,
                               DataType dtype,
                               size_t requested_size,
                               bool fake_alloc) {
  return value_->AllocateFrom(allocator, dtype, requested_size, fake_alloc);
}

const Place& DistTensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      value_->holder_,
      phi::errors::PreconditionNotMet(
          "Tensor not initialized yet when DenseTensor::place() is called."));
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
  meta_.dims = meta.dims;
  meta_.dtype = meta.dtype;
  meta_.is_scalar = meta.is_scalar;
  meta_.layout = meta.layout;
  meta_.lod = meta.lod;
  meta_.offset = meta.offset;
  meta_.use_gpudnn = meta.use_gpudnn;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
