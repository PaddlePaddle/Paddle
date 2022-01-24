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

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/stream/stream.h"

#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/storage.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

/* @jim19930609: Move to MKLDNN_Tensor in the future
    */
#ifdef PADDLE_WITH_MKLDNN
#include "dnnl.hpp"
#endif

namespace pten {

class CompatibleDenseTensorUtils;

/* --------------------------- */
/*   From framework::Tensor    */
/* --------------------------- */
/* Temporarily put TensorInplaceVersion inside DenseTensor.
   Will move to AutogradMeta as soon as we switch to Eager Dygraph.
   */
class TensorInplaceVersion {
 public:
  explicit TensorInplaceVersion(uint32_t inplace_version = 0)
      : inplace_version_(inplace_version) {}
  bool IsUnique() const { return inplace_version_ == 0; }
  void Bump() { ++inplace_version_; }
  uint32_t CurrentVersion() const { return inplace_version_; }
  void SetInplaceVersionToZero() { inplace_version_ = 0; }

 private:
  uint32_t inplace_version_;
};

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

  /// \brief Use existing storage space to create dense tensor. This interface
  /// can be used to deliberately create an uninitialized dense tensor.
  /// \param storage The existing storage.
  /// \param meta The meta data of dense tensor.
  DenseTensor(intrusive_ptr<Storage> storage, const DenseTensorMeta& meta);

  /// \brief Use existing storage space to create dense tensor. This interface
  /// can be used to deliberately create an uninitialized dense tensor.
  /// \param storage The existing storage.
  /// \param meta The meta data of dense tensor.
  DenseTensor(intrusive_ptr<Storage> storage, DenseTensorMeta&& meta);

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
  bool IsSharedWith(const DenseTensor& b) const;

  /// \brief Change the shape information in the metadata. If the new size is
  /// larger than the original value, the storage area will be reallocated.
  /// \param dims The new dims of the dense tensor.
  /// \param lod The new lod of the dense tensor.
  // void Resize(const DDim& dims);
  void ResizeAndAllocate(const DDim& dims);

  DenseTensor& Resize(const DDim& dims);

  /// \brief Change the lod information in the metadata.
  /// \param lod The new lod of the dense tensor.
  void ResetLoD(const LoD& lod);

  /// \brief Returns the actual storage size occupied by tensor, may be larger
  /// than its shape dims.
  /// \return The actual storage size occupied by tensor.
  size_t capacity() const { return storage_->size(); }

  /// \brief Get the mutable data pointer value of type T.
  /// Memory allocation may occur when calling this interface:
  /// 1. When the storage size is not enough to meet the current shape of the
  /// data.
  /// \return The mutable data pointer value of type T.
  template <typename T>
  T* mutable_data();

  /// \brief Get the mutable data pointer value of raw type.
  /// Memory allocation may occur when calling this interface:
  /// 1. When the storage size is not enough to meet the current shape of the
  /// data.
  /// 2. When more request_bytes parameters are used to reserve the data
  /// storage.
  /// param request_bytes The bytes to reserve the data storage.
  /// \return The mutable data pointer value of type T.
  void* mutable_data(size_t request_bytes = 0);

  /// \brief Get the const data pointer value of type T.
  /// \return The const data pointer value of type T.
  template <typename T>
  const T* data() const;

  /// \brief Get the const data pointer value of raw type.
  /// \return The const data pointer value of raw type.
  const void* data() const;

 private:
  friend class CompatibleDenseTensorUtils;

 protected:
  DenseTensorMeta meta_;
  intrusive_ptr<Storage> storage_;

  /* --------------------------- */
  /*   From framework::Tensor    */
  /* --------------------------- */
  /* The following members & interfaces were copied from framework::Tensor,
     so as to facilitate the unification of different Tensors

     Will be adjusted/removed/moved in the near future
   */
 public:
  /* @jim19930609: The way default constructor handles allocator might change,
     according to
                   the final design of Allocation - Allocator.
   */
  DenseTensor();

  /* @jim19930609: Remove dependency on protobuf after Tensor Unification.
   */
  explicit DenseTensor(const paddle::framework::proto::VarType::Type& dtype);

  inline bool IsInitialized() const {
    return storage_ != nullptr && storage_->data_shared() != nullptr;
  }

  template <typename T>
  T* data();

  void* data();

  template <typename T>
  T* mutable_data(const paddle::platform::Place& place,
                  size_t requested_size = 0);

  template <typename T>
  T* mutable_data(const DDim& dims,
                  const paddle::platform::Place& place,
                  size_t requested_size = 0);

  void* mutable_data(const paddle::platform::Place& place,
                     paddle::framework::proto::VarType::Type type,
                     size_t requested_size = 0);

  void* mutable_data(const paddle::platform::Place& place,
                     size_t requested_size = 0);

  void* mutable_data(const paddle::platform::Place& place,
                     paddle::framework::proto::VarType::Type type,
                     const paddle::platform::Stream& stream);

  /* @jim19930609: Remove dependency on protobuf after Tensor Unification.
   */
  paddle::framework::proto::VarType::Type type() const;

  /* @jim19930609: Remove dependency on protobuf after Tensor Unification.
   */
  paddle::framework::proto::VarType::Type saved_type() const;

  // memory size returns the holding memory size in byte.
  size_t memory_size() const;

  void check_memory_size() const;

  void set_layout(const paddle::framework::DataLayout layout);

  void clear() {
    storage_.reset();
    meta_.offset = 0;
  }

  void ShareBufferWith(const DenseTensor& tensor);

  void ShareDataTypeWith(const DenseTensor& tensor) {
    meta_.dtype = tensor.meta().dtype;
  }

  bool IsSharedBufferWith(const DenseTensor& src) const {
    if (storage_ == nullptr || src.storage_ == nullptr) return false;
    if (storage_->data_shared() == src.storage_->data_shared()) return true;

    return false;
  }

  const std::shared_ptr<paddle::memory::Allocation> Holder() const {
    return storage_ == nullptr ? nullptr : std::move(storage_->data_shared());
  }

  void set_offset(size_t offset) { meta_.offset = offset; }
  size_t offset() const { return meta_.offset; }

  std::shared_ptr<paddle::memory::Allocation> MoveMemoryHolder() {
    return storage_ == nullptr ? nullptr
                               : std::move(storage_->move_data_shared());
  }

  void ResetHolder(const std::shared_ptr<paddle::memory::Allocation>& holder);

  void ResetHolderWithType(
      const std::shared_ptr<paddle::memory::Allocation>& holder,
      const paddle::framework::proto::VarType::Type& type);

  void set_type(const paddle::framework::proto::VarType::Type& type);

  TensorInplaceVersion& InplaceVersionCounter() {
    return *inplace_version_counter_;
  }

  /*! The internal of two tensors share the same memory block. */
  DenseTensor& ShareDataWith(const DenseTensor& src);

  /*! The internal of two tensors share the same inplace version counter. */
  DenseTensor& ShareInplaceVersionCounterWith(const DenseTensor& src);

  DenseTensor Slice(int64_t begin_idx, int64_t end_idx) const;

  std::vector<DenseTensor> Split(int64_t split_size, int64_t axis) const;

  std::vector<DenseTensor> Chunk(int64_t chunks, int64_t axis) const;

 protected:
  std::shared_ptr<TensorInplaceVersion> inplace_version_counter_;

/* @jim19930609: This is a hack
   In general, it is badly designed to fuse MKLDNN-specific objects into a
   generic Tensor.
   We temporarily leave them here to unblock Tensor Unification progress.
   In the final state, we should come up with a MKLDNN_Tensor and move the
   following codes there.
   */
#ifdef PADDLE_WITH_MKLDNN

 public:
  inline dnnl::memory::format_tag format() const { return format_; }

  inline void set_format(const dnnl::memory::format_tag format) {
    format_ = format;
  }

 protected:
  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */

  dnnl::memory::format_tag format_ = dnnl::memory::format_tag::undef;
#endif

  /* ------------------------------ */
  /*   From framework::LoDTensor    */
  /* ------------------------------ */
  /* The following members & interfaces were copied from framework::Tensor,
     so as to facilitate the unification of different Tensors

     Will be adjusted/removed/moved in the near future
   */
 public:
  explicit DenseTensor(const LoD& lod);

  void set_lod(const LoD& lod);

  LoD* mutable_lod();

  /*
   * Get the start offset and end offset of an  element from LoD.
   */
  std::pair<size_t, size_t> lod_element(size_t level, size_t elem) const;

  size_t NumLevels() const;

  size_t NumElements(size_t level = 0) const;
};

}  // namespace pten
