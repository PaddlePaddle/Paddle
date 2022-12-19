/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/compat/convert_utils.h"

/**
 * [ Why still include the fluid headers? ]
 *
 * We hope to organize the basic implementation of Tensor and the logic related
 * to Tensor computation into an independent library, which we call
 * [Tensor Operation Library, phi], so we extract or rewrite the original
 * Kernels.
 *
 * In the future, the training library, inference library and custom operators
 * will link to this Tensor Operation library.
 *
 * However, if we directly split the link relation, we need to make too many
 * changes, which will affect the stability of the framework, so here we still
 * rely on the implementation of the framework, which is a intermediate state.
 *
 * In the future, the necessary components will be moved to the this library,
 * or the corresponding components will be re-implemented.
 */
#include "paddle/fluid/memory/malloc.h"

namespace phi {

DenseTensor::DenseTensor(Allocator* a, const DenseTensorMeta& meta)
    : meta_(meta), holder_(a->Allocate(SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(Allocator* a, DenseTensorMeta&& meta)
    : meta_(std::move(meta)), holder_(a->Allocate(SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(const std::shared_ptr<phi::Allocation>& holder,
                         const DenseTensorMeta& meta)
    : meta_(meta), holder_(holder) {}

DenseTensor::DenseTensor(const DenseTensor& other) : meta_(other.meta()) {
  holder_ = other.holder_;
  storage_properties_ =
      std::move(CopyStorageProperties(other.storage_properties_));
  inplace_version_counter_ = other.inplace_version_counter_;

#ifdef PADDLE_WITH_MKLDNN
  mem_desc_ = other.mem_desc_;
#endif
}

DenseTensor& DenseTensor::operator=(const DenseTensor& other) {
  meta_ = other.meta();
  holder_ = other.holder_;
  storage_properties_ =
      std::move(CopyStorageProperties(other.storage_properties_));
  inplace_version_counter_ = other.inplace_version_counter_;
#ifdef PADDLE_WITH_MKLDNN
  mem_desc_ = other.mem_desc_;
#endif
  return *this;
}

DenseTensor& DenseTensor::operator=(DenseTensor&& other) {
  meta_ = std::move(other.meta_);
  std::swap(holder_, other.holder_);
  storage_properties_ = std::move(other.storage_properties_);
  std::swap(inplace_version_counter_, other.inplace_version_counter_);
#ifdef PADDLE_WITH_MKLDNN
  mem_desc_ = other.mem_desc_;
#endif
  return *this;
}

int64_t DenseTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

bool DenseTensor::IsSharedWith(const DenseTensor& b) const {
  return holder_ && holder_ == b.Holder();
}

void* DenseTensor::AllocateFrom(Allocator* allocator,
                                DataType dtype,
                                size_t requested_size) {
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      phi::errors::InvalidArgument(
          "Required allocator shall not be nullptr, but received nullptr."));
  if (this->dtype() != dtype) {
    VLOG(10) << "change data type in mutbale_data, target dtype - " << dtype;
    meta_.dtype = dtype;
  }
  PADDLE_ENFORCE(
      valid(),
      phi::errors::PreconditionNotMet(
          "The meta data must be valid when call the mutable data function."));
  size_t bytes = numel() * SizeOf(this->dtype());
  if (requested_size) {
    PADDLE_ENFORCE_GE(requested_size,
                      bytes,
                      phi::errors::InvalidArgument(
                          "The reserved size %d should be enough to meet the "
                          "volume required by metadata %d.",
                          requested_size,
                          bytes));
    bytes = requested_size;
  }
  // NOTE(paddle-dev): In case of the allocator of storage_ is different with
  // the incoming allocator, we will re-alloc data using the incoming
  // allocator. See DeviceContext.Alloc in core/device_context.cc.
  if (!holder_ || holder_->size() < bytes + meta_.offset) {
    meta_.offset = 0;
    VLOG(10) << "Allocate data with bytes: " << bytes;
    auto holder = allocator->Allocate(bytes);
    if (holder_) {
      PADDLE_ENFORCE_LE(
          numel() * static_cast<int64_t>(SizeOf(dtype)) +
              static_cast<int64_t>(meta_.offset),
          static_cast<int64_t>(holder->size()),
          phi::errors::InvalidArgument(
              "The size of Holder is not enough to store the Tensor."));
    }
    holder_ = std::move(holder);
  }

  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 meta_.offset);
}

template <typename T>
const T* DenseTensor::data() const {
  PADDLE_ENFORCE_EQ(
      dtype(),
      paddle::experimental::CppTypeToDataType<T>::Type(),
      phi::errors::InvalidArgument(
          "The type of data we are trying to retrieve (%s) does not match the "
          "type of data (%s) currently contained in the container.",
          paddle::experimental::CppTypeToDataType<T>::Type(),
          dtype()));
  return static_cast<const T*>(data());
}

template <typename T>
T* DenseTensor::data() {
  T* ret = static_cast<T*>(data());
  PADDLE_ENFORCE(
      (dtype() == paddle::experimental::CppTypeToDataType<T>::Type()),
      phi::errors::InvalidArgument(
          "The type of data we are trying to retrieve (%s) does not match the "
          "type of data (%s) currently contained in the container.",
          paddle::experimental::CppTypeToDataType<T>::Type(),
          dtype()));
  return ret;
}

void* DenseTensor::data() {
  check_memory_size();
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      phi::errors::PreconditionNotMet(
          "The storage must be valid when call the data function."));
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 meta_.offset);
}

const void* DenseTensor::data() const {
  check_memory_size();
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      phi::errors::PreconditionNotMet(
          "The storage must be valid when call the data function."));
  return reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset);
}

void DenseTensor::set_meta(DenseTensorMeta&& meta) {
  PADDLE_ENFORCE(!meta_.valid(),
                 phi::errors::InvalidArgument(
                     "Only when the original attribute of Tensor is "
                     "incomplete, can it be reset."));
  meta_ = std::move(meta);
}

void DenseTensor::set_meta(const DenseTensorMeta& meta) {
  PADDLE_ENFORCE(
      meta.valid(),
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

/* @jim19930609: This interface will be further modified until we finalized the
   design for Allocator - Allocation
   For now, we have to temporarily accommodate two independent use cases:
   1. Designed behaviour: DenseTensor constructed with its underlying storage_
   initialized
   2. Legacy behaviour(fluid): DenseTensor constructed using default
   constructor, where
                               storage_ won't be initialized until the first
   call to mutable_data(place)
   */
void DenseTensor::ResizeAndAllocate(const DDim& dims) {
  meta_.dims = dims;
  if (holder_ != nullptr && place().GetType() != AllocationType::UNDEFINED) {
    mutable_data(place());
  }
}

void DenseTensor::ResetLoD(const LoD& lod) { meta_.lod = lod; }

#define DATA_MEMBER_FUNC_INSTANTIATION(dtype)      \
  template const dtype* DenseTensor::data() const; \
  template dtype* DenseTensor::data();

DATA_MEMBER_FUNC_INSTANTIATION(bool);
DATA_MEMBER_FUNC_INSTANTIATION(int8_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint8_t);
DATA_MEMBER_FUNC_INSTANTIATION(int16_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint16_t);
DATA_MEMBER_FUNC_INSTANTIATION(int32_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint32_t);
DATA_MEMBER_FUNC_INSTANTIATION(int64_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint64_t);
DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::bfloat16);
DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::float16);
DATA_MEMBER_FUNC_INSTANTIATION(float);
DATA_MEMBER_FUNC_INSTANTIATION(double);
DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::complex<float>);
DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::complex<double>);

#undef DATA_MEMBER_FUNC_INSTANTIATION

template <typename DeviceT>
const DeviceT& DenseTensor::storage_properties() const {
  PADDLE_ENFORCE_NOT_NULL(
      storage_properties_,
      phi::errors::PreconditionNotMet(
          "The storage_properties of current DenseTensor is nullptr."));
  if (DeviceT::classof(storage_properties_.get())) {
    return static_cast<DeviceT&>(*storage_properties_);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The actual type of storage_properties is inconsistent with the type "
        "of the template parameter passed in."));
  }
}

template const NPUStorageProperties& DenseTensor::storage_properties() const;
#ifdef PADDLE_WITH_MKLDNN
template const OneDNNStorageProperties& DenseTensor::storage_properties() const;
#endif

bool DenseTensor::storage_properties_initialized() const {
  return storage_properties_ != nullptr;
}

void DenseTensor::set_storage_properties(
    std::unique_ptr<StorageProperties>&& storage_properties) {
  storage_properties_ = std::move(storage_properties);
}

}  // namespace phi
