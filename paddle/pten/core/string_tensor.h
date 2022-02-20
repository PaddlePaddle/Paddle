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

#pragma once

#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/storage.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

/// \brief The Dense tensor store values in a contiguous sequential block
/// of memory where all values are represented. Tensors or multi-dimensional
/// arrays are used in math operators.
/// During the entire life cycle of a StringTensor, its device type and key
/// metadata are set unchanged.
class StringTensor : public TensorBase,
                     public TypeInfoTraits<TensorBase, StringTensor> {
 public:
  /// \brief Construct a dense tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  StringTensor(Allocator* a, const StringTensorMeta& meta);

  /// \brief Construct a dense tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  StringTensor(Allocator* a, StringTensorMeta&& meta);

  StringTensor(const std::shared_ptr<pten::Allocation>& holder,
               const StringTensorMeta& meta);

  /// \brief Because dense tensor is a resource handle, we provide a default
  /// move constructor to support move semantics.
  StringTensor(StringTensor&& other) = default;

  StringTensor(const StringTensor& other);

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~StringTensor() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "StringTensor"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override;

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override;

  /// \brief Returns the meta information of the tensor.
  /// \return The meta information of the tensor.
  const StringTensorMeta& meta() const noexcept { return meta_; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return DataType::STRING; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return DataLayout::STRINGS; }

  /// \brief Sets the meta information of the tensor. Only when the original
  /// attribute of Tensor is incomplete, can it be reset.
  /// \param meta The meta information of the tensor.
  void set_meta(StringTensorMeta&& meta);

  void set_meta(const StringTensorMeta& meta);

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept override { return meta_.valid(); }

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  bool initialized() const override { return holder_ && holder_->ptr(); }

  /// \brief Check if storage is shared with other objects.
  /// \return Whether the storage is shared with other objects.
  bool IsSharedWith(const StringTensor& b) const;

  const std::shared_ptr<pten::Allocation>& Holder() const { return holder_; }

  /// \brief Change the shape information in the metadata. If the new size is
  /// larger than the original value, the storage area will be reallocated.
  /// \param dims The new dims of the dense tensor.
  /// \param lod The new lod of the dense tensor.
  void ResizeAndAllocate(const DDim& dims);
  StringTensor& Resize(const DDim& dims);

  /// \brief Returns the actual storage size occupied by tensor, may be larger
  /// than its shape dims.
  /// \return The actual storage size occupied by tensor.
  size_t capacity() const { return holder_->size(); }

  /// \brief Get the mutable data pointer value of pstring type.
  /// Memory allocation may occur when calling this interface:
  /// 1. When the storage size is not enough to meet the current shape of the
  /// data.
  /// 2. When more request_bytes parameters are used to reserve the data
  /// storage.
  /// param request_bytes The bytes to reserve the data storage.
  /// \return The mutable data pointer value of type T.
  dtype::pstring* mutable_data(const paddle::platform::Place& place,
                               size_t request_bytes = 0);

  /// \brief Get the const data pointer value of pstring type.
  /// \return The const data pointer value of pstring type.
  const dtype::pstring* data() const;

  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0);

 private:
  StringTensorMeta meta_;
  std::shared_ptr<pten::Allocation> holder_;
  void init_holder();
};

}  // namespace pten
