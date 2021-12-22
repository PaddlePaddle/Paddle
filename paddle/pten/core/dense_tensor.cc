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

namespace pten {

DenseTensor::DenseTensor(const std::shared_ptr<Allocator>& a,
                         const DenseTensorMeta& meta)
    : meta_(meta),
      storage_(make_intrusive<TensorStorage>(a, SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(const std::shared_ptr<Allocator>& a,
                         DenseTensorMeta&& meta)
    : meta_(std::move(meta)),
      storage_(make_intrusive<TensorStorage>(a, SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(intrusive_ptr<Storage> storage,
                         const DenseTensorMeta& meta)
    : meta_(meta), storage_(std::move(storage)) {}

DenseTensor::DenseTensor(intrusive_ptr<Storage> storage, DenseTensorMeta&& meta)
    : meta_(std::move(meta)), storage_(std::move(storage)) {}

DenseTensor::DenseTensor(const DenseTensor& other)
    : meta_(other.meta()), storage_(copy_intrusive(other.storage_)) {}

int64_t DenseTensor::numel() const {
  if (meta_.is_scalar) {
    return 1;
  }
  return product(meta_.dims);
}

bool DenseTensor::IsSharedWith(const DenseTensor& b) const {
  return storage_.get() == b.storage_.get() && storage_.get() != nullptr;
}

void* DenseTensor::mutable_data(size_t request_bytes) {
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
  return storage_->data();
}

template <typename T>
T* DenseTensor::mutable_data() {
  // In order to be compatible with the original Tensor design and
  // execution system, we have to reset the datatype in mutable_data<T>.
  // When the compatibility phase is over in the future, we can delete it
  if (meta_.dtype == DataType::UNDEFINED) {
    VLOG(10) << "change data type in mutbale_data, target dtype - "
             << paddle::experimental::CppTypeToDataType<T>::Type();
    const_cast<DataType&>(meta_.dtype) =
        paddle::experimental::CppTypeToDataType<T>::Type();
  }
  PADDLE_ENFORCE(
      (dtype() == paddle::experimental::CppTypeToDataType<T>::Type()),
      paddle::platform::errors::InvalidArgument(
          "The type of data (%d) we are trying to retrieve does not match the "
          "type of data currently contained in the container (%d).",
          static_cast<int>(paddle::experimental::CppTypeToDataType<T>::Type()),
          static_cast<int>(dtype())));
  return static_cast<T*>(mutable_data());
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
  PADDLE_ENFORCE_NOT_NULL(
      storage_,
      paddle::platform::errors::PreconditionNotMet(
          "The storage must be valid when call the mutable data function."));
  return storage_->data();
}

void DenseTensor::set_meta(DenseTensorMeta&& meta) {
  PADDLE_ENFORCE(!meta_.valid(),
                 paddle::platform::errors::InvalidArgument(
                     "Only when the original attribute of Tensor is "
                     "incomplete, can it be reset."));
  meta_ = std::move(meta);
}

void DenseTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  mutable_data();
}

void DenseTensor::ResetLoD(const LoD& lod) { meta_.lod = lod; }

#define DATA_MEMBER_FUNC_INSTANTIATION(dtype)  \
  template dtype* DenseTensor::mutable_data(); \
  template const dtype* DenseTensor::data() const;

DATA_MEMBER_FUNC_INSTANTIATION(bool);
DATA_MEMBER_FUNC_INSTANTIATION(int8_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint8_t);
DATA_MEMBER_FUNC_INSTANTIATION(int16_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint16_t);
DATA_MEMBER_FUNC_INSTANTIATION(int32_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint32_t);
DATA_MEMBER_FUNC_INSTANTIATION(int64_t);
DATA_MEMBER_FUNC_INSTANTIATION(uint64_t);
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::platform::bfloat16);
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::platform::float16);
DATA_MEMBER_FUNC_INSTANTIATION(float);
DATA_MEMBER_FUNC_INSTANTIATION(double);
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::experimental::complex64);
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::experimental::complex128);

#undef DATA_MEMBER_FUNC_INSTANTIATION

}  // namespace pten
