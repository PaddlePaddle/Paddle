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

#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type.h"

#include "paddle/phi/core/tensor_base.h"

/* @jim19930609: Move to MKLDNN_Tensor in the future
    */
#ifdef PADDLE_WITH_MKLDNN
#include "dnnl.hpp"
#endif

namespace phi {

/// \brief The Vocab Tensor store values in a contiguous sequential block
/// of memory where all values are represented. Tensors or multi-dimensional
/// arrays are used in math operators.
/// During the entire life cycle of a VocabTensor, its device type and key
/// metadata are set unchanged.
class VocabTensor : public TensorBase,
                    public TypeInfoTraits<TensorBase, VocabTensor> {
 private:
  std::unordered_map<std::string, int32_t> data_;
  DDim* dim_;
  phi::CPUPlace cpu_place_;

 public:
  /// \brief Construct a vocab tensor and allocate space.
  /// \param data The vocab data.
  explicit VocabTensor(const std::unordered_map<std::string, int32_t>& data);

  /// \brief Because vocab tensor is a kind of container, we give a default
  /// constructor to use for stl container. But the vocab tensor created with
  /// the default constructor is not practical.
  VocabTensor() = default;

  /// \brief Because vocab tensor is a resource handle, we provide a default
  /// move constructor to support move semantics.
  VocabTensor(VocabTensor&& other);

  /// \brief VocabTensor shallow copy constructor.
  VocabTensor(const VocabTensor& other);

  /// \brief VocabTensor deep copy assignment.
  VocabTensor& operator=(const VocabTensor& other);

  /// \brief VocabTensor shallow copy assignment.
  VocabTensor& operator=(VocabTensor&& other);

  /// \brief Destroy the tensor object and release exclusive resources.
  ~VocabTensor() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "VocabTensor"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const { return data_.size(); }

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const { return *dim_; }

  /// \brief Returns the data of the tensor.
  /// \return The data of the tensor.
  const std::unordered_map<std::string, int32_t>& data() const { return data_; }

  /// \brief Returns the data type of the resource tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept {
    return paddle::experimental::DataType::UNDEFINED;
  }

  /// \brief Returns the key data type of the resource tensor.
  /// \return The data type of the tensor.
  DataType key_type() const noexcept {
    return paddle::experimental::DataType::INT64;
  }

  /// \brief Returns the value data type of the resource tensor.
  /// \return The data type of the tensor.
  DataType value_type() const noexcept {
    return paddle::experimental::DataType::INT32;
  }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const {
    return paddle::experimental::DataLayout::UNDEFINED;
  }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const { return cpu_place_; }

  /// \brief Always valid.
  bool valid() const noexcept { return true; }

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  bool initialized() const { return data_.size() > 0; }

  /// \brief Check if storage is shared with other objects.
  /// \return Whether the data_ is shared with other objects.
  bool IsSharedWith(const VocabTensor& b) const;

  /// \brief Returns the actual map size occupied by tensor.
  /// \return The actual map size occupied by tensor.
  size_t capacity() const { return data_.size(); }

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0) override {
    return nullptr;
  }

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
};

}  // namespace phi
