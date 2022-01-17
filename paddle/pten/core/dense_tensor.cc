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

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/convert_utils.h"

namespace pten {

DenseTensor::DenseTensor(Allocator* a, const DenseTensorMeta& meta)
    : meta_(meta), holder_(a->Allocate(SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(Allocator* a, DenseTensorMeta&& meta)
    : meta_(std::move(meta)), holder_(a->Allocate(SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(intrusive_ptr<Storage> storage,
                         const DenseTensorMeta& meta)
    : meta_(meta), holder_(storage->move_data_shared()) {
  if (meta.valid()) {
    mutable_data(storage->place());
  }
}

DenseTensor::DenseTensor(intrusive_ptr<Storage> storage, DenseTensorMeta&& meta)
    : meta_(std::move(meta)), holder_(storage->move_data_shared()) {
  if (meta.valid()) {
    mutable_data(storage->place());
  }
}

DenseTensor::DenseTensor(const DenseTensor& other) : meta_(other.meta()) {
  holder_ = other.holder_;
#ifdef PADDLE_WITH_MKLDNN
  format_ = other.format_;
#endif
}

DenseTensor& DenseTensor::operator=(const DenseTensor& other) {
  meta_ = other.meta();
  holder_ = other.holder_;
#ifdef PADDLE_WITH_MKLDNN
  format_ = other.format_;
#endif
  return *this;
}

DenseTensor& DenseTensor::operator=(DenseTensor&& other) {
  meta_ = std::move(other.meta_);
  std::swap(holder_, other.holder_);
  return *this;
}

int64_t DenseTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

template <typename T>
const T* DenseTensor::data() const {
  PADDLE_ENFORCE(
      (dtype() == paddle::experimental::CppTypeToDataType<T>::Type()),
      paddle::platform::errors::InvalidArgument(
          "The type of data we are trying to retrieve does not match the "
          "type of data currently contained in the container."));
  return static_cast<const T*>(data());
}

const void* DenseTensor::data() const {
  check_memory_size();
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      paddle::platform::errors::PreconditionNotMet(
          "The storage must be valid when call the data function."));
  return reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset);
}

void DenseTensor::set_meta(DenseTensorMeta&& meta) {
  PADDLE_ENFORCE(!meta_.valid(),
                 paddle::platform::errors::InvalidArgument(
                     "Only when the original attribute of Tensor is "
                     "incomplete, can it be reset."));
  meta_ = std::move(meta);
}

DenseTensor& DenseTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  return *this;
}

void DenseTensor::ResetLoD(const LoD& lod) { meta_.lod = lod; }

/* --------------------------- */
/*   From framework::Tensor    */
/* --------------------------- */
DenseTensor::DenseTensor() {
  inplace_version_counter_ = std::make_shared<TensorInplaceVersion>(0);
  meta_.dtype = paddle::experimental::DataType::FLOAT32;
  meta_.offset = 0;
}

DenseTensor::DenseTensor(const paddle::framework::proto::VarType::Type& dtype) {
  inplace_version_counter_ = std::make_shared<TensorInplaceVersion>(0);
  meta_.dtype = TransToPtenDataType(dtype);
  meta_.offset = 0;
}

size_t DenseTensor::memory_size() const {
  return holder_ == nullptr ? 0UL : holder_->size() - meta_.offset;
}

void DenseTensor::check_memory_size() const {
  PADDLE_ENFORCE_NOT_NULL(holder_,
                          paddle::platform::errors::PreconditionNotMet(
                              "Tensor holds no memory. "
                              "Call Tensor::mutable_data firstly."));
  PADDLE_ENFORCE_LE(
      numel() * SizeOf(dtype()),
      memory_size(),
      paddle::platform::errors::PreconditionNotMet(
          "Tensor's dimension is out of bound."
          "Tensor's dimension must be equal or less than the size of its "
          "memory."
          "But received Tensor's dimension is d%, memory's size is %d.",
          numel() * SizeOf(dtype()),
          memory_size()));
}

const paddle::platform::Place& DenseTensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      paddle::platform::errors::PreconditionNotMet(
          "Tensor not initialized yet when DenseTensor::place() is called."));
  return holder_->place();
}

paddle::framework::proto::VarType::Type DenseTensor::type() const {
  return TransToProtoVarType(meta_.dtype);
}

paddle::framework::proto::VarType::Type DenseTensor::saved_type() const {
  return TransToProtoVarType(meta_.dtype);
}

void DenseTensor::set_layout(const paddle::framework::DataLayout layout) {
  meta_.layout = layout;
}

void DenseTensor::ResetHolder(
    const std::shared_ptr<paddle::memory::Allocation>& holder) {
  PADDLE_ENFORCE_EQ(
      meta_.offset,
      0,
      paddle::platform::errors::Fatal(
          "Only the offset is supported to zero when the holder is reset."));
  if (holder_) {
    PADDLE_ENFORCE_LE(
        numel() * SizeOf(dtype()) + meta_.offset,
        holder->size(),
        paddle::platform::errors::InvalidArgument(
            "The size of Holder is not enough to store the Tensor."));
  }
  holder_ = holder;
}

void DenseTensor::ResetHolderWithType(
    const std::shared_ptr<paddle::memory::Allocation>& holder,
    const paddle::framework::proto::VarType::Type& type) {
  set_type(type);
  ResetHolder(holder);
}

void DenseTensor::set_type(
    const paddle::framework::proto::VarType::Type& type) {
  meta_.dtype = TransToPtenDataType(type);
}

void* DenseTensor::mutable_data(const paddle::platform::Place& place,
                                paddle::framework::proto::VarType::Type type,
                                size_t requested_size) {
  meta_.dtype = TransToPtenDataType(type);
  PADDLE_ENFORCE_GE(
      numel(),
      0,
      paddle::platform::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(),
          "] now"));
  size_t size = numel() * SizeOf(dtype());
  if (requested_size) {
    PADDLE_ENFORCE_GE(
        requested_size,
        size,
        paddle::platform::errors::InvalidArgument(
            "The requested memory size is less than the memory size of Tensor. "
            "But received requested memory size is %d, "
            "memory size of Tensor is %d.",
            requested_size,
            size));
    size = requested_size;
  }
  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + meta_.offset) {
    // Reset holder first before re-allocate to save memory
    holder_.reset();
    holder_ = paddle::memory::AllocShared(place, size);
    meta_.offset = 0;
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 meta_.offset);
}

void* DenseTensor::mutable_data(const paddle::platform::Place& place,
                                size_t requested_size) {
  return mutable_data(place, type(), requested_size);
}

void* DenseTensor::mutable_data(const paddle::platform::Place& place,
                                paddle::framework::proto::VarType::Type type,
                                const paddle::platform::Stream& stream) {
  set_type(type);
  PADDLE_ENFORCE_GE(
      numel(),
      0,
      paddle::platform::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(),
          "] now"));
  size_t size = numel() * SizeOf(dtype());

  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + meta_.offset ||
      !(paddle::platform::is_gpu_place(place) &&
        paddle::memory::InSameStream(holder_, stream))) {
    holder_.reset();
    holder_ = paddle::memory::AllocShared(place, size, stream);
    meta_.offset = 0;
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 meta_.offset);
}

/* @jim19930609: The following "mutable_data" only supports specific dtypes
   defined in OpProto. This part need another clean up once the data type across
   Fluid
   and Pten get unified.
   */
template <typename T>
inline T* DenseTensor::mutable_data(const DDim& dims,
                                    const paddle::platform::Place& place,
                                    size_t requested_size) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  Resize(dims);
  return mutable_data<T>(place, requested_size);
}

template <typename T>
inline T* DenseTensor::mutable_data(const paddle::platform::Place& place,
                                    size_t requested_size) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  return reinterpret_cast<T*>(mutable_data(
      place, paddle::framework::DataTypeTrait<T>::DataType(), requested_size));
}

void DenseTensor::ShareBufferWith(const DenseTensor& tensor) {
  holder_ = tensor.holder_;
  meta_.offset = tensor.meta().offset;
  meta_.dtype = tensor.dtype();
}

template <typename T>
T* DenseTensor::data() {
  PADDLE_ENFORCE(
      (dtype() == paddle::experimental::CppTypeToDataType<T>::Type()),
      paddle::platform::errors::InvalidArgument(
          "The type of data we are trying to retrieve does not match the "
          "type of data currently contained in the container."));
  return static_cast<T*>(data());
}

void* DenseTensor::data() {
  check_memory_size();
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      paddle::platform::errors::PreconditionNotMet(
          "The storage must be valid when call the data function."));
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 meta_.offset);
}

#define LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(dtype) \
  template const dtype* DenseTensor::data() const;   \
  template dtype* DenseTensor::data();               \
  template dtype* DenseTensor::mutable_data(         \
      const DDim& dims,                              \
      const paddle::platform::Place& place,          \
      size_t requested_size);                        \
  template dtype* DenseTensor::mutable_data(         \
      const paddle::platform::Place& place, size_t requested_size);

LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(bool)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int8_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(uint8_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int16_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int32_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int64_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(float)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(double)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::paddle::platform::bfloat16)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::paddle::platform::float16)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::paddle::experimental::complex64)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::paddle::experimental::complex128)

#undef LEGACY_DATA_MEMBER_FUNC_INSTANTIATION

/* ------------------------------ */
/*   From framework::LoDTensor    */
/* ------------------------------ */

DenseTensor::DenseTensor(const LoD& lod) : DenseTensor() { meta_.lod = lod; }

void DenseTensor::set_lod(const LoD& lod) { meta_.lod = lod; }

LoD* DenseTensor::mutable_lod() { return &meta_.lod; }

std::pair<size_t, size_t> DenseTensor::lod_element(size_t level,
                                                   size_t elem) const {
  PADDLE_ENFORCE_LT(
      level,
      NumLevels(),
      paddle::platform::errors::InvalidArgument(
          "The input level of LoD is invalid, it should be less than LoD "
          "size. The input level is %zu, the LoD size is %zu.",
          level,
          NumLevels()));

  PADDLE_ENFORCE_LT(elem,
                    NumElements(level),
                    paddle::platform::errors::InvalidArgument(
                        "The input element of LoD is invalid, it should be "
                        "less than the number of elements in its level."
                        "The input element is %zu, the number of elements in "
                        "its level is %zu.",
                        elem,
                        NumElements(level)));

  return std::make_pair((meta_.lod)[level][elem], (meta_.lod)[level][elem + 1]);
}

size_t DenseTensor::NumLevels() const { return meta_.lod.size(); }

size_t DenseTensor::NumElements(size_t level) const {
  PADDLE_ENFORCE_LT(
      level,
      NumLevels(),
      paddle::platform::errors::InvalidArgument(
          "The input level of LoD is invalid, it should be less than LoD "
          "size. The input level is %zu, the LoD size is %zu.",
          level,
          NumLevels()));

  // the last offset is the end of last element
  return (meta_.lod)[level].size() - 1;
}

}  // namespace pten
