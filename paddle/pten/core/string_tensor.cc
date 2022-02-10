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
#include "paddle/fluid/memory/malloc.h"

namespace pten {

StringTensor::StringTensor(Allocator* a, const StringTensorMeta& meta)
    : meta_(meta), holder_(a->Allocate(SizeOf(dtype()) * numel())) {
  init_holder();
}

StringTensor::StringTensor(Allocator* a, StringTensorMeta&& meta)
    : meta_(std::move(meta)), holder_(a->Allocate(SizeOf(dtype()) * numel())) {
  init_holder();
}

StringTensor::StringTensor(const std::shared_ptr<pten::Allocation>& holder,
                           const StringTensorMeta& meta)
    : meta_(meta), holder_(holder) {}

int64_t StringTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

bool StringTensor::IsSharedWith(const StringTensor& b) const {
  return holder_ && holder_ == b.Holder();
}

const Place& StringTensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      paddle::platform::errors::PreconditionNotMet(
          "Tensor not initialized yet when DenseTensor::place() is called."));
  return holder_->place();
}

dtype::pstring* StringTensor::mutable_data(const paddle::platform::Place& place,
                                           size_t request_bytes /* = 0 */) {
  PADDLE_ENFORCE_GE(
      numel(),
      0,
      paddle::platform::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(),
          "] now"));
  size_t size = numel() * SizeOf(dtype());
  if (request_bytes && (request_bytes > size)) {
    size = request_bytes;
  }

  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + meta_.offset) {
    holder_.reset();
    holder_ = paddle::memory::AllocShared(place, size);
    // Initialize the allocated bytes
    init_holder();
    meta_.offset = 0;
  }
  return reinterpret_cast<dtype::pstring*>(
      reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset);
}

const dtype::pstring* StringTensor::data() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      paddle::platform::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  return reinterpret_cast<dtype::pstring*>(
      reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset);
}

void StringTensor::set_meta(StringTensorMeta&& meta) {
  PADDLE_ENFORCE(!meta_.valid(),
                 paddle::platform::errors::InvalidArgument(
                     "Only when the original attribute of Tensor is "
                     "incomplete, can it be reset."));
  meta_ = std::move(meta);
}

void StringTensor::set_meta(const StringTensorMeta& meta) {
  PADDLE_ENFORCE(
      meta.valid(),
      paddle::platform::errors::InvalidArgument(
          "Input meta is invalid, please check the meta attribute."));
  meta_.dims = meta.dims;
  meta_.is_scalar = meta.is_scalar;
  meta_.offset = meta.offset;
}

void StringTensor::ResizeAndAllocate(const DDim& dims) {
  meta_.dims = dims;
  if (holder_ != nullptr && place().GetType() != AllocationType::UNDEFINED) {
    mutable_data(place());
  }
}

StringTensor& StringTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  return *this;
}
// TODO(zhoushunjie): need to remove it for general space
void StringTensor::init_holder() {
  void* ptr = holder_->ptr();
  auto& place = holder_->place();
  auto bytes_size = holder_->size();
  if (paddle::platform::is_cpu_place(place)) {
    std::memset(ptr, 0, bytes_size);
  } else if (paddle::platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_HIP
    hipMemset(ptr, 0, bytes_size);
#else
    cudaMemset(ptr, 0, bytes_size);
#endif
  }
}
void* StringTensor::AllocateFrom(Allocator* allocator,
                                 DataType dtype,
                                 size_t requested_size) {
  // TODO(zhoushunjie): implement it later.
  return nullptr;
}

}  // namespace pten
