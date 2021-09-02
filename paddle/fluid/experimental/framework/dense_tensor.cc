/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/experimental/framework/dense_tensor.h"

namespace paddle {
namespace experimental {
namespace framework {

DenseTensorMeta::DenseTensorMeta(DataType type, const DDim& dims)
    : dims(dims), type(type) {}
DenseTensorMeta::DenseTensorMeta(DataType type, const DDim& dims,
                                 DataLayout layout)
    : dims(dims), type(type), layout(layout) {}
DenseTensorMeta::DenseTensorMeta(DataType type, const DDim& dims,
                                 DataLayout layout,
                                 const std::vector<std::vector<size_t>>& lod)
    : dims(dims), type(type), layout(layout), lod(lod) {}

bool DenseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (type != DataType::INVALID);
  valid = valid && (layout != DataLayout::Undef);
  valid = valid && (is_scalar || product(dims));
  return valid;
}

DenseTensor::DenseTensor(const std::shared_ptr<Allocator>& a,
                         const DenseTensorMeta& meta)
    : meta_(meta),
      storage_(make_intrusive<Storage>(a, SizeOf(data_type()) * numel())) {}

DenseTensor::DenseTensor(const std::shared_ptr<Allocator>& a,
                         DenseTensorMeta&& meta)
    : meta_(std::move(meta)),
      storage_(make_intrusive<Storage>(a, SizeOf(data_type()) * numel())) {}

DenseTensor::DenseTensor(intrusive_ptr<Storage>&& storage,
                         const DenseTensorMeta& meta)
    : meta_(meta), storage_(std::move(storage)) {}

DenseTensor::DenseTensor(intrusive_ptr<Storage>&& storage,
                         DenseTensorMeta&& meta)
    : meta_(std::move(meta)), storage_(std::move(storage)) {}

int64_t DenseTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

bool DenseTensor::SharesStorageWith(const DenseTensor& b) const {
  return storage_.get() == b.storage_.get() && storage_.get() != nullptr;
}

template <typename T>
T* DenseTensor::mutable_data(size_t request_bytes) {
  PADDLE_ENFORCE(
      valid(),
      platform::errors::PreconditionNotMet(
          "The meta data must be valid when call the mutable data function."));
  PADDLE_ENFORCE_NOT_NULL(
      storage_,
      platform::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  PADDLE_ENFORCE(
      meta_.type == DataTypeTrait<T>::data_type(),
      platform::errors::PreconditionNotMet(
          "The type of data we are trying to retrieve does not match the "
          "type of data currently contained in the container."));
  size_t bytes = numel() * SizeOf(data_type());
  if (request_bytes) {
    PADDLE_ENFORCE_GE(request_bytes, bytes,
                      paddle::platform::errors::InvalidArgument(
                          "The reserved size %d should be enough to meet the "
                          "volume required by metadata %d.",
                          request_bytes, bytes));
    bytes = request_bytes;
  }
  if (storage_->size() < bytes) {
    storage_->Realloc(bytes);
  }
  return static_cast<T*>(storage_->data());
}

template <typename T>
const T* DenseTensor::data() const {
  PADDLE_ENFORCE_NOT_NULL(
      storage_,
      platform::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  PADDLE_ENFORCE(
      meta_.type == DataTypeTrait<T>::data_type(),
      platform::errors::PreconditionNotMet(
          "The type of data we are trying to retrieve does not match the "
          "type of data currently contained in the container."));
  return static_cast<const T*>(storage_->data());
}

void DenseTensor::check_memory_size() const {
  size_t bytes = numel() * SizeOf(data_type());
  PADDLE_ENFORCE_GE(memory_size(), bytes,
                    paddle::platform::errors::InvalidArgument(
                        "The memory size %d should be enough to meet the "
                        "volume required by metadata %d.",
                        memory_size(), bytes));
}

#define DATA_MEMBER_FUNC_INSTANTIATION(dtype)                      \
  template dtype* DenseTensor::mutable_data(size_t request_bytes); \
  template const dtype* DenseTensor::data() const;

DATA_MEMBER_FUNC_INSTANTIATION(int8_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint8_t);
DATA_MEMBER_FUNC_INSTANTIATION(int16_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint16_t);
DATA_MEMBER_FUNC_INSTANTIATION(int32_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint32_t);
DATA_MEMBER_FUNC_INSTANTIATION(int64_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint64_t);
DATA_MEMBER_FUNC_INSTANTIATION(float);
DATA_MEMBER_FUNC_INSTANTIATION(double);

#undef DATA_MEMBER_FUNC_INSTANTIATION

}  // namespace framework
}  // namespace experimental
}  // namespace paddle
