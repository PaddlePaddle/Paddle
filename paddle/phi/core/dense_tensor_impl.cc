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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
/* --------------------------- */
/*   From phi::DenseTensor     */
/* --------------------------- */
DenseTensor::DenseTensor() {
  meta_.dtype = phi::DataType::FLOAT32;
  meta_.offset = 0;
}

DenseTensor::DenseTensor(phi::DataType dtype) {
  meta_.dtype = dtype;
  meta_.offset = 0;
}

size_t DenseTensor::memory_size() const {
  return holder_ == nullptr ? 0UL : holder_->size() - meta_.offset;
}

void DenseTensor::check_memory_size() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      common::errors::PreconditionNotMet("Tensor holds no memory. "
                                         "Call Tensor::mutable_data firstly."));
  if (meta_.is_contiguous()) {
    PADDLE_ENFORCE_LE(
        numel() * SizeOf(dtype()),
        memory_size(),
        common::errors::PreconditionNotMet(
            "Tensor's dimension is out of bound."
            "Tensor's dimension must be equal or less than the size of its "
            "memory."
            "But received Tensor's dimension is %d, memory's size is %d.",
            numel() * SizeOf(dtype()),
            memory_size()));
  }
}

const Place& DenseTensor::place() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_,
      common::errors::PreconditionNotMet(
          "Tensor not initialized yet when DenseTensor::place() is called."));
  return holder_->place();
}

phi::DataType DenseTensor::type() const { return meta_.dtype; }

void DenseTensor::set_layout(const DataLayout layout) {
  if (meta_.strides.size() == -1) {
    meta_.strides = meta_.calc_strides(meta_.dims);
  }
  meta_.layout = layout;
}

// Note: When you reset holder, you need to ensure the offset is correct
void DenseTensor::ResetHolder(const std::shared_ptr<phi::Allocation>& holder) {
  if (holder_ && meta_.is_contiguous()) {
    PADDLE_ENFORCE_LE(
        numel() * static_cast<int64_t>(SizeOf(dtype())) +
            static_cast<int64_t>(meta_.offset),
        static_cast<int64_t>(holder->size()),
        common::errors::InvalidArgument(
            "The size of Holder is not enough to store the Tensor."));
  }
  holder_ = holder;
}

void DenseTensor::ResetHolderWithType(
    const std::shared_ptr<phi::Allocation>& holder, phi::DataType type) {
  set_type(type);
  ResetHolder(holder);
}

void DenseTensor::set_type(phi::DataType type) { meta_.dtype = type; }

void* DenseTensor::mutable_data(const Place& place,
                                phi::DataType type,
                                size_t requested_size) {
  set_type(type);
  PADDLE_ENFORCE_GE(
      numel(),
      0,
      common::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(),
          "] now"));
  size_t size = numel() * SizeOf(dtype());
  if (requested_size && (requested_size > size)) {
    size = requested_size;
  }

  /* some versions of paddle::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + meta_.offset) {
    holder_.reset();
    holder_ = memory_utils::AllocShared(place, size);
    meta_.offset = 0;
  }
  uintptr_t ptr = reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset;
  return reinterpret_cast<void*>(ptr);
}

void* DenseTensor::mutable_data(const Place& place, size_t requested_size) {
  return mutable_data(place, type(), requested_size);
}

void* DenseTensor::mutable_data(const Place& place,
                                phi::DataType type,
                                const phi::Stream& stream) {
  set_type(type);
  PADDLE_ENFORCE_GE(
      numel(),
      0,
      common::errors::PreconditionNotMet(
          "The Tensor's element number must be equal or greater than zero. "
          "The Tensor's shape is [",
          dims(),
          "] now"));
  size_t size = numel() * SizeOf(dtype());

  /* some versions of paddle::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + meta_.offset ||
      !(place.GetType() == phi::AllocationType::GPU &&
        memory_utils::InSameStream(holder_, stream))) {
    holder_.reset();
    holder_ = memory_utils::AllocShared(place, size, stream);
    meta_.offset = 0;
  }
  uintptr_t ptr = reinterpret_cast<uintptr_t>(holder_->ptr()) + meta_.offset;
  return reinterpret_cast<void*>(ptr);
}

/* @jim19930609: The following "mutable_data" only supports specific dtypes
   defined in OpProto. This part need another clean up once the data type across
   Fluid
   and Phi get unified.
   */
template <typename T>
inline T* DenseTensor::mutable_data(const DDim& dims,
                                    const Place& place,
                                    size_t requested_size) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  if (meta_.dims.size() != -1 && meta_.dims != dims) {
    PADDLE_ENFORCE_EQ(meta_.is_contiguous(),
                      true,
                      common::errors::InvalidArgument(
                          "Right now Resize is only supported for contiguous "
                          "Tensor. Tensor dims is %s, Tensor layout is %s, "
                          "Tensor stride is %s. New dims is %s.",
                          meta_.dims,
                          meta_.layout,
                          meta_.strides,
                          dims));
  }
  meta_.dims = dims;
  meta_.strides = meta_.calc_strides(meta_.dims);
  return mutable_data<T>(place, requested_size);
}

template <typename T>
inline T* DenseTensor::mutable_data(const Place& place, size_t requested_size) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  return reinterpret_cast<T*>(
      mutable_data(place, phi::CppTypeToDataType<T>::Type(), requested_size));
}

void DenseTensor::ShareBufferWith(const DenseTensor& tensor, bool only_buffer) {
  holder_ = tensor.holder_;
  if (!only_buffer) {
    meta_.offset = tensor.meta().offset;
    meta_.dtype = tensor.dtype();
  }
}

#define LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(dtype)                     \
  template TEST_API dtype* DenseTensor::mutable_data(                    \
      const DDim& dims, const Place& place, size_t requested_size);      \
  template TEST_API dtype* DenseTensor::mutable_data(const Place& place, \
                                                     size_t requested_size);

LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(bool)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int8_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(uint8_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int16_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(uint16_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int32_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(uint32_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(int64_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(uint64_t)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::bfloat16)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::float16)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::float8_e4m3fn)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::float8_e5m2)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(float)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(double)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::complex<float>)
LEGACY_DATA_MEMBER_FUNC_INSTANTIATION(::phi::dtype::complex<double>)

#undef LEGACY_DATA_MEMBER_FUNC_INSTANTIATION

/* ------------------------------ */
/*   From phi::DenseTensor    */
/* ------------------------------ */

DenseTensor::DenseTensor(const LegacyLoD& legacy_lod) : DenseTensor() {
  meta_.legacy_lod = legacy_lod;
}

void DenseTensor::set_lod(const LegacyLoD& legacy_lod) {
  meta_.legacy_lod = legacy_lod;
}

LoD* DenseTensor::mutable_lod() { return &meta_.legacy_lod; }

std::pair<size_t, size_t> DenseTensor::lod_element(size_t level,
                                                   size_t elem) const {
  PADDLE_ENFORCE_LT(
      level,
      NumLevels(),
      common::errors::InvalidArgument(
          "The input level of LoD is invalid, it should be less than LoD "
          "size. The input level is %zu, the LoD size is %zu.",
          level,
          NumLevels()));

  PADDLE_ENFORCE_LT(elem,
                    NumElements(level),
                    common::errors::InvalidArgument(
                        "The input element of LoD is invalid, it should be "
                        "less than the number of elements in its level."
                        "The input element is %zu, the number of elements in "
                        "its level is %zu.",
                        elem,
                        NumElements(level)));

  return std::make_pair((meta_.legacy_lod)[level][elem],
                        (meta_.legacy_lod)[level][elem + 1]);
}

size_t DenseTensor::NumLevels() const { return meta_.legacy_lod.size(); }

size_t DenseTensor::NumElements(size_t level) const {
  PADDLE_ENFORCE_LT(
      level,
      NumLevels(),
      common::errors::InvalidArgument(
          "The input level of LoD is invalid, it should be less than LoD "
          "size. The input level is %zu, the LoD size is %zu.",
          level,
          NumLevels()));

  // the last offset is the end of last element
  return (meta_.legacy_lod)[level].size() - 1;
}

DenseTensor& DenseTensor::Resize(const DDim& dims) {
  if (meta_.dims.size() != -1 && meta_.dims != dims) {
    PADDLE_ENFORCE_EQ(meta_.is_contiguous(),
                      true,
                      common::errors::InvalidArgument(
                          "Right now Resize is only supported for contiguous "
                          "Tensor. Tensor dims is %s, Tensor layout is %s, "
                          "Tensor stride is %s. New dims is %s.",
                          meta_.dims,
                          meta_.layout,
                          meta_.strides,
                          dims));
  }
  meta_.dims = dims;
  meta_.strides = meta_.calc_strides(meta_.dims);
  return *this;
}

DenseTensor DenseTensor::Slice(int64_t begin_idx, int64_t end_idx) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(
      begin_idx,
      0,
      common::errors::OutOfRange("The start row index must be greater than 0."
                                 "But received the start index is d%.",
                                 begin_idx));
  PADDLE_ENFORCE_LE(
      end_idx,
      meta_.dims[0],
      common::errors::OutOfRange("The end row index is out of bound."));
  PADDLE_ENFORCE_LT(
      begin_idx,
      end_idx,
      common::errors::InvalidArgument(
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
                    common::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));

  PADDLE_ENFORCE_GE(
      split_size,
      0,
      common::errors::OutOfRange(
          "split expects split_size be non-negative, but got split_size is %d",
          split_size));

  int64_t numel_size = meta_.dims[static_cast<int>(axis)];

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
                    common::errors::OutOfRange(
                        "split expects at least a 1-dimensional tensor"));
  PADDLE_ENFORCE_GE(
      chunks,
      0,
      common::errors::OutOfRange(
          "chunks expects to be greater than 0, but got chunks is %d", chunks));

  int64_t numel_size = meta_.dims[static_cast<int>(axis)];
  int64_t split_size = (numel_size + chunks - 1) / chunks;
  return Split(split_size, axis);
}

#ifdef PADDLE_WITH_DNNL
const dnnl::memory::desc& DenseTensor::mem_desc() const {
  if (storage_properties_ == nullptr) {
    static dnnl::memory::desc undef_desc = dnnl::memory::desc();
    return undef_desc;
  }
  return this->storage_properties<OneDNNStorageProperties>().mem_desc;
}

void DenseTensor::set_mem_desc(const dnnl::memory::desc& mem_desc) {
  if (storage_properties_ == nullptr) {
    storage_properties_ = std::make_unique<OneDNNStorageProperties>();
    static_cast<OneDNNStorageProperties*>(storage_properties_.get())->mem_desc =
        mem_desc;
    meta_.layout = DataLayout::ONEDNN;
  } else if (OneDNNStorageProperties::classof(storage_properties_.get())) {
    static_cast<OneDNNStorageProperties*>(storage_properties_.get())->mem_desc =
        mem_desc;
    meta_.layout = DataLayout::ONEDNN;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The actual type of storage_properties is inconsistent with the type "
        "of the template parameter passed in."));
  }
}
#endif

// NOTE: For historical reasons, this interface has a special behavior,
// sharing other tensor members except lod
DenseTensor& DenseTensor::ShareDataWith(const DenseTensor& src) {
  src.check_memory_size();
  holder_ = src.holder_;
  meta_.is_scalar = src.meta_.is_scalar;
  meta_.dims = src.meta_.dims;
  meta_.dtype = src.meta_.dtype;
  meta_.layout = src.meta_.layout;
  meta_.offset = src.meta_.offset;
  meta_.use_gpudnn = src.meta_.use_gpudnn;
  meta_.strides = src.meta_.strides;
  storage_properties_ = CopyStorageProperties(src.storage_properties_);
  return *this;
}

DenseTensor& DenseTensor::ShareDataNoCheckWith(const DenseTensor& src) {
  holder_ = src.holder_;
  set_meta(src.meta());
  storage_properties_ = CopyStorageProperties(src.storage_properties_);
  return *this;
}

DenseTensor& DenseTensor::ShareInplaceVersionCounterWith(
    const DenseTensor& src) {
  PADDLE_ENFORCE_NOT_NULL(
      inplace_version_counter_,
      common::errors::PreconditionNotMet(
          "Tensor does not hold inplace_version_counter_."));

  inplace_version_counter_ = src.inplace_version_counter_;
  return *this;
}
}  // namespace phi
