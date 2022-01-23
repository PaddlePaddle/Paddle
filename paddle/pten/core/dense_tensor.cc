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
#include "paddle/pten/common/bfloat16.h"
#include "paddle/pten/common/complex.h"
#include "paddle/pten/common/float16.h"

#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/convert_utils.h"

namespace paddle {
namespace framework {
extern void TensorCopy(const pten::DenseTensor& src,
                       const paddle::platform::Place& dst_place,
                       pten::DenseTensor* dst);
}
}

namespace pten {

DenseTensor::DenseTensor(Allocator* a, const DenseTensorMeta& meta)
    : meta_(meta), holder_(a->Allocate(SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(Allocator* a, DenseTensorMeta&& meta)
    : meta_(std::move(meta)), holder_(a->Allocate(SizeOf(dtype()) * numel())) {}

DenseTensor::DenseTensor(
    const std::shared_ptr<paddle::memory::Allocation>& holder,
    const DenseTensorMeta& meta)
    : meta_(meta), holder_(holder) {}

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

bool DenseTensor::IsSharedWith(const DenseTensor& b) const {
  return holder_ && holder_ == b.Holder();
}

template <typename T>
const T* DenseTensor::data() const {
  check_memory_size();
  PADDLE_ENFORCE(
      (dtype() == paddle::experimental::CppTypeToDataType<T>::Type()),
      paddle::platform::errors::InvalidArgument(
          "The type of data we are trying to retrieve does not match the "
          "type of data currently contained in the container."));
  return static_cast<const T*>(data());
}

template <typename T>
T* DenseTensor::data() {
  check_memory_size();
  PADDLE_ENFORCE(
      (dtype() == paddle::experimental::CppTypeToDataType<T>::Type()),
      paddle::platform::errors::InvalidArgument(
          "The type of data we are trying to retrieve does not match the "
          "type of data currently contained in the container."));
  return static_cast<T*>(data());
}

void* DenseTensor::data() {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      paddle::platform::errors::PreconditionNotMet(
          "The storage must be valid when call the data function."));
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 meta_.offset);
}

const void* DenseTensor::data() const {
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

/* @jim19930609: This interface will be further modified util we finalized the
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
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::platform::bfloat16);
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::platform::float16);
DATA_MEMBER_FUNC_INSTANTIATION(float);
DATA_MEMBER_FUNC_INSTANTIATION(double);
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::experimental::complex64);
DATA_MEMBER_FUNC_INSTANTIATION(::paddle::experimental::complex128);

#undef DATA_MEMBER_FUNC_INSTANTIATION

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
  if (requested_size && (requested_size > size)) {
    size = requested_size;
  }

  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + meta_.offset) {
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
  meta_.dims = dims;
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

#define LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(dtype) \
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

DenseTensor::DenseTensor(intrusive_ptr<Storage> storage,
                         const DenseTensorMeta& meta)
    : meta_(meta), holder_(storage->move_data_shared()) {}

DenseTensor::DenseTensor(intrusive_ptr<Storage> storage, DenseTensorMeta&& meta)
    : meta_(std::move(meta)), holder_(storage->move_data_shared()) {}

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

DenseTensor& DenseTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  return *this;
}

DenseTensor DenseTensor::Slice(int64_t begin_idx, int64_t end_idx) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(begin_idx,
                    0,
                    paddle::platform::errors::OutOfRange(
                        "The start row index must be greater than 0."
                        "But received the start index is d%.",
                        begin_idx));
  PADDLE_ENFORCE_LE(end_idx,
                    meta_.dims[0],
                    paddle::platform::errors::OutOfRange(
                        "The end row index is out of bound."));
  PADDLE_ENFORCE_LT(
      begin_idx,
      end_idx,
      paddle::platform::errors::InvalidArgument(
          "The start row index must be less than the end row index."
          "But received the start index = %d, the end index = %d.",
          begin_idx,
          end_idx));

  if (meta_.dims[0] == 1) {
    return *this;
  } else {
    size_t base = numel() / meta_.dims[0];
    DenseTensor dst;
    dst.holder_ = holder_;
    dst.set_layout(meta_.layout);
    dst.meta_.dtype = meta_.dtype;
    DDim dst_dims = meta_.dims;
    dst_dims[0] = end_idx - begin_idx;
    dst.Resize(dst_dims);
    dst.meta_.offset = meta_.offset + begin_idx * base * SizeOf(dtype());
    return dst;
  }
}

std::vector<DenseTensor> DenseTensor::Split(int64_t split_size,
                                            int64_t axis) const {
  check_memory_size();

  PADDLE_ENFORCE_GE(meta_.dims.size(),
                    0,
                    paddle::platform::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));

  PADDLE_ENFORCE_GE(
      split_size,
      0,
      paddle::platform::errors::OutOfRange(
          "split expects split_size be non-negative, but got split_size is %d",
          split_size));

  int64_t numel_size = meta_.dims[axis];

  int64_t num_splits = 1;
  if (split_size != 0) {
    num_splits =
        std::max<int64_t>((numel_size + split_size - 1) / split_size, 1);
  }

  std::vector<DenseTensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - numel_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    int64_t length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = Slice(i * split_size, i * split_size + length);
  }
  return splits;
}

std::vector<DenseTensor> DenseTensor::Chunk(int64_t chunks,
                                            int64_t axis) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(meta_.dims.size(),
                    0,
                    paddle::platform::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));
  PADDLE_ENFORCE_GE(
      chunks,
      0,
      paddle::platform::errors::OutOfRange(
          "chunks expects to be greater than 0, but got chunks is %d", chunks));

  int64_t numel_size = meta_.dims[axis];
  int64_t split_size = (numel_size + chunks - 1) / chunks;
  return Split(split_size, axis);
}

DenseTensor& DenseTensor::ShareDataWith(const DenseTensor& src) {
  src.check_memory_size();
  // Preserve LoD
  auto lod = meta_.lod;
  *this = src;
  meta_.lod = lod;
  return *this;
}

DenseTensor& DenseTensor::ShareInplaceVersionCounterWith(
    const DenseTensor& src) {
  PADDLE_ENFORCE_NOT_NULL(
      inplace_version_counter_,
      paddle::platform::errors::PreconditionNotMet(
          "Tensor does not hold inplace_version_counter_."));

  inplace_version_counter_ = src.inplace_version_counter_;
  return *this;
}

}  // namespace pten
