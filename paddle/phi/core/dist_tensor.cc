/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/dist_tensor.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

DistTensor::DistTensor() { local_tensor_ = std::make_shared<DenseTensor>(); }

/// \brief Construct a dist tensor and allocate space.
/// \param a The allocator used to allocate space.
/// \param meta The meta data of dense tensor.
DistTensor::DistTensor(Allocator* a, const DenseTensorMeta& meta) {
  local_tensor_ = std::make_shared<DenseTensor>(a, meta);
}

DistTensor::DistTensor(const std::shared_ptr<phi::Allocation>& holder,
                       const DenseTensorMeta& meta) {
  local_tensor_ = std::make_shared<DenseTensor>(holder, meta);
}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& dense_tensor)
    : local_tensor_(dense_tensor) {}

int64_t DistTensor::numel() const { return local_tensor_->numel(); }

const DDim& DistTensor::dims() const { return local_tensor_->dims(); }

DataType DistTensor::dtype() const { return local_tensor_->dtype(); }

/// \brief Returns the data layout of the tensor.
/// \return The data layout of the tensor.
DataLayout DistTensor::layout() const { return local_tensor_->layout(); }

/// \brief Returns the data place of the tensor.
/// \return The data place of the tensor.
const Place& DistTensor::place() const { return local_tensor_->place(); }

/// \brief Test whether the metadata is valid.
/// \return Whether the metadata is valid.
bool DistTensor::valid() const { return local_tensor_->valid(); }

/// \brief Test whether the storage is allocated.
/// \return Whether the storage is allocated.
bool DistTensor::initialized() const { return local_tensor_->initialized(); }
// TODO(Aurelius84): This interface is under intermediate state now.
// We will remove DataType argument in the future. Please DO NOT
// rely on Datatype too much when designing and implementing other features.

/// \brief Allocate memory with requested size from allocator.
/// \return The mutable data pointer value of type T.
void* DistTensor::AllocateFrom(Allocator* allocator,
                               DataType dtype,
                               size_t requested_size,
                               bool fake_alloc) {
  return local_tensor_->AllocateFrom(
      allocator, dtype, requested_size, fake_alloc);
}

void DistTensor::set_meta(DenseTensorMeta&& meta) {
  local_tensor_->set_meta(std::move(meta));
}

void DistTensor::set_meta(const DenseTensorMeta& meta) {
  local_tensor_->set_meta(meta);
}

const DenseTensorMeta& DistTensor::meta() const noexcept {
  return local_tensor_->meta();
}

const std::shared_ptr<DistTensor::DistAttr>& DistTensor::get_dist_attr() {
  return dist_attr_;
}

const std::shared_ptr<DenseTensor>& DistTensor::local_tensor() const {
  return local_tensor_;
}
}  // namespace phi
