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

#include "paddle/pten/core/string_tensor.h"

namespace pten {

StringTensor::StringTensor(Allocator* a, const StringTensorMeta& meta)
    : meta_(meta),
      storage_(make_intrusive<TensorStorage>(a, SizeOf(dtype()) * numel())) {}

StringTensor::StringTensor(Allocator* a, StringTensorMeta&& meta)
    : meta_(std::move(meta)),
      storage_(make_intrusive<TensorStorage>(a, SizeOf(dtype()) * numel())) {}

StringTensor::StringTensor(intrusive_ptr<Storage> storage,
                           const StringTensorMeta& meta)
    : meta_(meta), storage_(std::move(storage)) {}

StringTensor::StringTensor(intrusive_ptr<Storage> storage,
                           StringTensorMeta&& meta)
    : meta_(std::move(meta)), storage_(std::move(storage)) {}

int64_t StringTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

bool StringTensor::IsSharedWith(const StringTensor& b) const {
  return storage_.get() == b.storage_.get() && storage_.get() != nullptr;
}

dtype::pstring* StringTensor::mutable_data(size_t request_bytes /* = 0 */) {
  PADDLE_ENFORCE(
      valid(),
      paddle::platform::errors::PreconditionNotMet(
          "The meta data must be valid when call the mutable data function."));
  PADDLE_ENFORCE_NOT_NULL(
      storage_,
      paddle::platform::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  size_t bytes = numel() * SizeOf(dtype());
  if (request_bytes) {
    PADDLE_ENFORCE_GE(request_bytes,
                      bytes,
                      paddle::platform::errors::InvalidArgument(
                          "The reserved size %d should be enough to meet the "
                          "volume required by metadata %d.",
                          request_bytes,
                          bytes));
    bytes = request_bytes;
  }
  if (storage_->size() < bytes || storage_->size() == 0) {
    VLOG(10) << "mutbale data realloc, original size: " << storage_->size()
             << ", new size: " << bytes;
    storage_->Realloc(bytes);
  }
  return reinterpret_cast<dtype::pstring*>(storage_->data());
}

const dtype::pstring* StringTensor::data() const {
  PADDLE_ENFORCE_NOT_NULL(
      storage_,
      paddle::platform::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  return reinterpret_cast<dtype::pstring*>(storage_->data());
}

void StringTensor::set_meta(StringTensorMeta&& meta) {
  PADDLE_ENFORCE(!meta_.valid(),
                 paddle::platform::errors::InvalidArgument(
                     "Only when the original attribute of Tensor is "
                     "incomplete, can it be reset."));
  meta_ = std::move(meta);
}

void StringTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  mutable_data();
}

}  // namespace pten
