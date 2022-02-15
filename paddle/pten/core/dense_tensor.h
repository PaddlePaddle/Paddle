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

#pragma once

#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/storage.h"
#include "paddle/pten/core/stream.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/data_type.h"

/* @jim19930609: Move to MKLDNN_Tensor in the future
    */
#ifdef PADDLE_WITH_MKLDNN
#include "dnnl.hpp"
#endif

namespace pten {

class DenseTensorUtils;

/// \brief The Dense tensor store values in a contiguous sequential block
/// of memory where all values are represented. Tensors or multi-dimensional
/// arrays are used in math operators.
/// During the entire life cycle of a DenseTensor, its device type and key
/// metadata are set unchanged.
class DenseTensor : public TensorBase,
                    public TypeInfoTraits<TensorBase, DenseTensor> {
 public:
  /// \brief Construct a dense tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  DenseTensor(Allocator* a, const DenseTensorMeta& meta);

  /// \brief Construct a dense tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  DenseTensor(Allocator* a, DenseTensorMeta&& meta);

  DenseTensor(const std::shared_ptr<pten::Allocation>& holder,
              const DenseTensorMeta& meta);

  /// \brief Because dense tensor is a kind of container, we give a default
  /// constructor to use for stl container. But the dense tensor created with
  /// the default constructor is not practical.
  // DenseTensor() = default;

  /// \brief Because dense tensor is a resource handle, we provide a default
  /// move constructor to support move semantics.
  DenseTensor(DenseTensor&& other) = default;

  /// \brief DenseTensor shallow copy constructor.
  DenseTensor(const DenseTensor& other);

  /// \brief DenseTensor shallow copy assignment.
  DenseTensor& operator=(const DenseTensor& other);

  DenseTensor& operator=(DenseTensor&& other);

  DenseTensor();

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~DenseTensor() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "DenseTensor"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override;

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the lod of the tensor.
  /// \return The lod of the tensor.
  const LoD& lod() const noexcept { return meta_.lod; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return meta_.dtype; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return meta_.layout; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override;

  /// \brief Returns the meta information of the tensor.
  /// \return The meta information of the tensor.
  const DenseTensorMeta& meta() const noexcept { return meta_; }

  /// \brief Sets the meta information of the tensor. Only when the original
  /// attribute of Tensor is incomplete, can it be reset.
  /// \param meta The meta information of the tensor.
  void set_meta(DenseTensorMeta&& meta);

  void set_meta(const DenseTensorMeta& meta);

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept override { return meta_.valid(); }

  /// \brief Test whether the allocation is allocated.
  /// return Whether the allocation is allocated.
  bool initialized() const override { return holder_ && holder_->ptr(); }

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0) override;

  /// \brief Check if allocation is shared with other objects.
  /// \return Whether the allocation is shared with other objects.
  bool IsSharedWith(const DenseTensor& b) const;

  /// \brief Change the shape information in the metadata. If the new size is
  /// larger than the original value, the allocation area will be reallocated.
  /// \param dims The new dims of the dense tensor.
  /// \param lod The new lod of the dense tensor.
  // void Resize(const DDim& dims);
  void ResizeAndAllocate(const DDim& dims);

  DenseTensor& Resize(const DDim& dims);

  /// \brief Change the lod information in the metadata.
  /// \param lod The new lod of the dense tensor.
  void ResetLoD(const LoD& lod);

  /// \brief Returns the actual allocation size occupied by tensor, may be
  /// larger
  /// than its shape dims.
  /// \return The actual allocation size occupied by tensor.
  size_t capacity() const { return holder_->size(); }

  /// \brief Get the const data pointer value of type T.
  /// \return The const data pointer value of type T.
  template <typename T>
  const T* data() const;

  /// \brief Get the const data pointer value of raw type.
  /// \return The const data pointer value of raw type.
  const void* data() const;

  template <typename T>
  T* data();

  void* data();

 private:
  friend class DenseTensorUtils;

 protected:
  DenseTensorMeta meta_;
  std::shared_ptr<pten::Allocation> holder_;

#include "paddle/pten/core/dense_tensor.inl"
};
}  // namespace pten
