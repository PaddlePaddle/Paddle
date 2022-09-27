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

#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/stream.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

/* @jim19930609: Move to MKLDNN_Tensor in the future
 */
#ifdef PADDLE_WITH_MKLDNN
#include "dnnl.hpp"  // NOLINT
#endif

namespace phi {

class DenseTensorUtils;

/// \brief The Dense tensor stores values in a contiguous sequential block
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

  DenseTensor(const std::shared_ptr<phi::Allocation>& holder,
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
  std::shared_ptr<phi::Allocation> holder_;

 public:
  /* Temporarily put InplaceVersion inside DenseTensor.
  Will move to AutogradMeta as soon as we switch to Eager Dygraph.
  */
  /*
  NOTE(liym27): [ What is TensorInplaceVersion used for? ]

  TensorInplaceVersion is a version counter and every Tensor has a version
  counter. It's used to check whether an inplace operation will result in an
  incorrect gradient calculation. Version is incremented when the data of the
  Variable is modified in place.

  - Question: In what scenarios will version counters be shared?
  - Answer: When two Variables/VarBases share the same C++ Tensor(its Allocation
  may change), both of them share the same version counter. For examples:
   1. `z = paddle.assign(input=x, output=y)`, `z` shares the same version
  counter of `y` because z and y is the same VarBase;
   2. `y = x.detach()`, `y` shares the same version counter of `x`.

  - Question: In what scenarios will version counters NOT be shared?
  - Answer: Replacing a `Variable`'s data by calling
  `Tensor::ShareDataWith(...)` or `Tensor::ShareBufferWith(...)`. Because they
  share the same Allocation but not framework::Tensor.

  - Question: Why put the inplace_version_counter_ in framework::Tensor instead
  of Allocation or Variable?
  - Answer:
   1. Tensor can call ResetHolder() to reset the corresponding Allocation so
  that the inplace_version_counter_ changes if it's in Allocation, which will
  lead to confusing information about inplace version.
   2. If inplace_version_counter_ is in Variable, different VariableWrappers
   should be able to share the same Variable. However, a VariableWrapper hold a
   Variable object but not a pointer.
 */
  class InplaceVersion {
   public:
    bool IsUnique() const { return inplace_version_ == 0; }
    void Bump() { ++inplace_version_; }
    uint32_t CurrentVersion() const { return inplace_version_; }
    void SetInplaceVersionToZero() { inplace_version_ = 0; }

   private:
    uint32_t inplace_version_{0};
  };

 protected:
  std::shared_ptr<InplaceVersion> inplace_version_counter_{
      std::make_shared<InplaceVersion>()};

/* @jim19930609: This is a hack
In general, it is badly designed to fuse MKLDNN-specific objects into a
generic Tensor.
We temporarily leave them here to unblock Tensor Unification progress.
In the final state, we should come up with a MKLDNN_Tensor and move the
following codes there.
*/
#ifdef PADDLE_WITH_MKLDNN
  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */
  dnnl::memory::format_tag format_ = dnnl::memory::format_tag::undef;

  /// \brief memory descriptor of tensor which have layout set as kMKLDNN
  dnnl::memory::desc mem_desc_;
#endif

#ifndef PADDLE_WITH_CUSTOM_KERNEL
#include "paddle/phi/core/dense_tensor.inl"
#endif
};

}  // namespace phi
