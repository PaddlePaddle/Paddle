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

#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/platform/pstring.h"
#include "paddle/pten/core/storage.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

class CompatibleDenseTensorUtils;

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
  StringTensor(const std::shared_ptr<Allocator>& a,
               const StringTensorMeta& meta);

  /// \brief Construct a dense tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  StringTensor(const std::shared_ptr<Allocator>& a, StringTensorMeta&& meta);

  /// \brief Use existing storage space to create dense tensor. This interface
  /// can be used to deliberately create an uninitialized dense tensor.
  /// \param storage The existing storage.
  /// \param meta The meta data of dense tensor.
  StringTensor(intrusive_ptr<Storage> storage, const StringTensorMeta& meta);

  /// \brief Use existing storage space to create dense tensor. This interface
  /// can be used to deliberately create an uninitialized dense tensor.
  /// \param storage The existing storage.
  /// \param meta The meta data of dense tensor.
  StringTensor(intrusive_ptr<Storage> storage, StringTensorMeta&& meta);

  /// \brief Because dense tensor is a kind of container, we give a default
  /// constructor to use for stl container. But the dense tensor created with
  /// the default constructor is not practical.
  StringTensor() = default;

  /// \brief Because dense tensor is a resource handle, we provide a default
  /// move constructor to support move semantics.
  StringTensor(StringTensor&& other) = default;

  /// \brief We do not recommend deep copy of dense tensor because of its
  /// efficiency and complexity across devices. The operation is disabled here.
  StringTensor(const StringTensor& other) = delete;

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
  const Place& place() const override { return storage_->place(); }

  /// \brief Returns the meta information of the tensor.
  /// \return The meta information of the tensor.
  const StringTensorMeta& meta() const noexcept { return meta_; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return DataType::STRING; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return DataLayout::ALL_LAYOUT; }

  /// \brief Sets the meta information of the tensor. Only when the original
  /// attribute of Tensor is incomplete, can it be reset.
  /// \param meta The meta information of the tensor.
  void set_meta(StringTensorMeta&& meta);

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept override { return meta_.valid(); }

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  bool initialized() const override {
    return storage_ != nullptr && storage_->data() != nullptr;
  }

  /// \brief Check if storage is shared with other objects.
  /// \return Whether the storage is shared with other objects.
  bool IsSharedWith(const StringTensor& b) const;

  /// \brief Change the shape information in the metadata. If the new size is
  /// larger than the original value, the storage area will be reallocated.
  /// \param dims The new dims of the dense tensor.
  /// \param lod The new lod of the dense tensor.
  void Resize(const DDim& dims);

  /// \brief Returns the actual storage size occupied by tensor, may be larger
  /// than its shape dims.
  /// \return The actual storage size occupied by tensor.
  size_t capacity() const { return storage_->size(); }

  /// \brief Release the storage area for other purposes. Because of the
  /// destruction of encapsulation, we do not support two dense tensors directly
  /// sharing the same intrusive pointer.
  /// \return The rvalue of instrusize pointer releated to the released storage.
  intrusive_ptr<Storage> release() { return std::move(storage_); }

  /// \brief Get the mutable data pointer value of pstring type.
  /// Memory allocation may occur when calling this interface:
  /// 1. When the storage size is not enough to meet the current shape of the
  /// data.
  /// 2. When more request_bytes parameters are used to reserve the data
  /// storage.
  /// param request_bytes The bytes to reserve the data storage.
  /// \return The mutable data pointer value of type T.
  platform::pstring* mutable_data(size_t request_bytes = 0);

  /// \brief Get the const data pointer value of pstring type.
  /// \return The const data pointer value of pstring type.
  const platform::pstring* data() const;

 private:
  friend class CompatibleDenseTensorUtils;

 private:
  StringTensorMeta meta_;
  intrusive_ptr<Storage> storage_;
};

}  // namespace pten
