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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/convert_utils.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace pten {

using CPUPlace = paddle::platform::CPUPlace;
using CUDAPlace = paddle::platform::CUDAPlace;
using CUDAPinnedPlace = paddle::platform::CUDAPinnedPlace;
using XPUPlace = paddle::platform::XPUPlace;
using NPUPlace = paddle::platform::NPUPlace;
using NPUPinnedPlace = paddle::platform::NPUPinnedPlace;

const paddle::platform::Place& DenseTensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      allocation_,
      paddle::platform::errors::PreconditionNotMet(
          "Tensor not initialized yet when Tensor::place() is called."));
  return allocation_->place();
}

//----------------------------------------------------------------
// Inner methods

void DenseTensor::ShareAllocation(
    const std::shared_ptr<paddle::memory::allocation::Allocation>& allocation) {
  // This operation can be very slow!
  // std::shared_ptr reference count is atomic. increasing or decreasing
  // the reference count requires atomic increment or decrement.
  // This is hundred times slower than non-atomic increment/decrement
  allocation_ = allocation;
}

// TODO(chenweihang): Add other place branchs
paddle::platform::Place DenseTensor::GetPlaceByBackend() const {
  switch (backend_) {
    case Backend::CPU:
      return CPUPlace();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case Backend::CUDA:
      return CUDAPlace(paddle::platform::GetCurrentDeviceId());
#endif
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported Tensor backend."));
  }
}

size_t DenseTensor::MemorySize() const {
  return allocation_ == nullptr ? 0UL : allocation_->size() - offset_;
}

void DenseTensor::CheckMemorySize() const {
  PADDLE_ENFORCE_NOT_NULL(allocation_,
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

const void* DenseTensor::data() const {
  CheckMemorySize();
  return reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(allocation_->ptr()) + offset_);
}

void* DenseTensor::mutable_data() {
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
  if (allocation_ == nullptr) {
    allocation_.reset();
    allocation_ = paddle::memory::AllocShared(place, size);
  } else {
    if (!(allocation_->place() == place) ||
        allocation_->size() < size + offset_) {
      allocation_.reset();
      allocation_ = paddle::memory::AllocShared(place, size);
    } else {
      // do nothing
    }
  }
  return reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(allocation_->ptr()) + offset_);
}

}  // namespace pten
