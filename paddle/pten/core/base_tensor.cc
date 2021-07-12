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

#include "paddle/pten/core/base_tensor.h"
#include "paddle/pten/core/convert_utils.h"

// fluid headers [may be replaced by new impl]
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace pt {

// TODO(chenweihang): Place still link to framework, design abstract interface
// of place?
using CPUPlace = paddle::platform::CPUPlace;
using CUDAPlace = paddle::platform::CUDAPlace;
using CUDAPinnedPlace = paddle::platform::CUDAPinnedPlace;
using XPUPlace = paddle::platform::XPUPlace;
using NPUPlace = paddle::platform::NPUPlace;
using NPUPinnedPlace = paddle::platform::NPUPinnedPlace;

BaseTensor::BaseTensor(TensorMeta meta)
    : meta_(std::forward<TensorMeta>(meta)) {}

int64_t BaseTensor::numel() const { return product(meta_.dims); }

DDim BaseTensor::dims() const { return meta_.dims; }

void BaseTensor::resize(const DDim& dims) { meta_.dims = dims; }

DataType BaseTensor::type() const { return meta_.type; }

Layout BaseTensor::layout() const { return meta_.layout; }

Place BaseTensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      memory_,
      paddle::platform::errors::PreconditionNotMet(
          "Tensor not initialized yet when Tensor::place() is called."));
  return memory_->place();
}

Backend BaseTensor::backend() const { return meta_.backend; }

bool BaseTensor::initialized() const { return memory_ != nullptr; }

//----------------------------------------------------------------
// Inner methods

void BaseTensor::ShareAllocation(const std::shared_ptr<Allocation>& memory) {
  // This operation can be very slow!
  // std::shared_ptr reference count is atomic. increasing or decreasing
  // the reference count requires atomic increment or decrement.
  // This is hundred times slower than non-atomic increment/decrement
  memory_ = memory;
}

// TODO(chenweihang): Add other place branchs
Place BaseTensor::GetPlaceByBackend() const {
  switch (meta_.backend) {
    case Backend::kCPU:
      return CPUPlace();
    case Backend::kCUDA:
      return CUDAPlace();
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported Tensor backend."));
  }
}

size_t BaseTensor::MemorySize() const {
  return memory_ == nullptr ? 0UL : memory_->size() - meta_.offset;
}

void BaseTensor::CheckMemorySize() const {
  PADDLE_ENFORCE_NOT_NULL(memory_,
                          paddle::platform::errors::PreconditionNotMet(
                              "Tensor holds no memory. "
                              "Call Tensor::mutable_data firstly."));
  size_t size_of_type =
      paddle::framework::SizeOfType(TransToProtoVarType(meta_.type));
  PADDLE_ENFORCE_LE(
      numel() * size_of_type,
      MemorySize(),
      paddle::platform::errors::PreconditionNotMet(
          "Tensor's dimension is out of bound."
          "Tensor's dimension must be equal or less than the size of its "
          "memory."
          "But received  Tensor's dimension is d%, memory's size is %d.",
          numel() * size_of_type,
          MemorySize()));
}

std::shared_ptr<Allocation> BaseTensor::MoveMemory() {
  return std::move(memory_);
}

const void* BaseTensor::data() const {
  CheckMemorySize();
  return reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(memory_->ptr()) + meta_.offset);
}

void* BaseTensor::mutable_data() {
  PADDLE_ENFORCE_GE(
      numel(),
      0,
      paddle::platform::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(),
          "] now"));
  size_t size =
      numel() * paddle::framework::SizeOfType(TransToProtoVarType(meta_.type));
  auto place = GetPlaceByBackend();
  if (memory_ == nullptr) {
    memory_.reset();
    memory_ = paddle::memory::AllocShared(place, size);
  } else {
    LOG(WARNING) << "When call mutable_data, BaseTensor has been initialized.";
    if (!(memory_->place() == place) || memory_->size() < size + meta_.offset) {
      memory_.reset();
      memory_ = paddle::memory::AllocShared(place, size);
    } else {
      // do nothing
    }
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(memory_->ptr()) +
                                 meta_.offset);
}

}  // namespace pt
