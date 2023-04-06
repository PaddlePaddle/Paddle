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

#include "paddle/phi/core/distributed/dist_tensor.h"

namespace phi {
int64_t DistTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

const DDim& DistTensor::dims() const { return meta_.dims; }

DataType DistTensor::dtype() const { return meta_.dtype; }

/// \brief Returns the data layout of the tensor.
/// \return The data layout of the tensor.
DataLayout DistTensor::layout() const { return meta_.layout; }

/// \brief Returns the data place of the tensor.
/// \return The data place of the tensor.
const Place& DistTensor::place() const { return place_; }

/// \brief Test whether the metadata is valid.
/// \return Whether the metadata is valid.
bool DistTensor::valid() const { return meta_.valid(); }

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
                               size_t requested_size = 0,
                               bool fake_alloc = false) {}

void DistTensor::set_meta(DenseTensorMeta&& meta) { std::swap(meta_, meta); }

void DistTensor::set_meta(const DenseTensorMeta& meta) { meta_ = meta; }

const DistAttr& DistTensor::get_dist_attr() { return dist_attr_; }
}  // namespace phi
