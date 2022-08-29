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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class TensorArray : public TensorBase,
                    public TypeInfoTraits<TensorBase, TensorArray> {
 public:
  /// \brief Construct a TensorArray.
  /// \param vec The vector DenseTensor used to init TensorArray.
  explicit TensorArray(const std::vector<DenseTensor>& vec);

  explicit TensorArray(size_t n) { tensors_.resize(n); }

  TensorArray() = default;

  TensorArray(TensorArray&& other) = default;

  TensorArray(const TensorArray& other) = default;

  /// \brief TensorArray shallow copy assignment.
  TensorArray& operator=(const TensorArray& other) = default;

  TensorArray& operator=(TensorArray&& other) = default;

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~TensorArray() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "TensorArray"; }

  /// \brief This overrided function is not used in TensorArray.
  /// \return It will return -1.
  int64_t numel() const override { return -1; }

  /// \brief Returns the dims of the tensor in TensorArray.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the data place of the tensor in TensorArray.
  /// \return The data place of the tensor.
  const Place& place() const override { return place_; }

  /// \brief Returns the data type of the tensor in TensorArray.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return meta_.dtype; }

  /// \brief Returns the data layout of the tensor in TensorArray.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return meta_.layout; }

  /// \brief This overrided function is not used in TensorArray.
  /// \return false.
  bool valid() const noexcept override { return false; }

  /// \brief Test whether the tensor's storage in TensorArray is allocated.
  /// return Whether all tensors in TensorArray is allocated.
  bool initialized() const override;

  /// \brief Clear all tensors in TensorArray.
  void clear() { tensors_.clear(); }

  /// \brief Allocate memory with requested size for all tensors from allocator.
  /// \return Void pointer
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0);

  bool empty() const { return tensors_.empty(); }

  /// \brief Returns the number of tensors in TensorArray.
  size_t size() const { return tensors_.size(); }

  /// \brief Resizes the TensorArray so that it contains n tensors.
  void resize(size_t n) { tensors_.resize(n); }

  /// \brief Requests that the TensorArray capacity be at least enough to
  /// contain n tensors.
  void reserve(size_t n) { tensors_.reserve(n); }

  /// \brief Add the tensor to the end of TensorArray
  void push_back(const DenseTensor& tensor);

  void emplace_back();

  void emplace_back(const DenseTensor& tensor);

  /// \brief Return the last tensor in TensorArray
  DenseTensor& back() { return tensors_.back(); }

  DenseTensor& at(size_t index) { return tensors_.at(index); }

  const DenseTensor& at(size_t index) const { return tensors_.at(index); }

  const DenseTensor& operator[](size_t index) const { return tensors_[index]; }

  DenseTensor& operator[](size_t index) { return tensors_[index]; }

  std::vector<DenseTensor>::iterator begin() { return tensors_.begin(); }

  std::vector<DenseTensor>::const_iterator begin() const {
    return tensors_.begin();
  }

  std::vector<DenseTensor>::iterator end() { return tensors_.end(); }

  std::vector<DenseTensor>::const_iterator end() const {
    return tensors_.end();
  }

 private:
  std::vector<DenseTensor> tensors_;
  // meta_ store the tensor's meta if all tensors in TensorArray have same meta.
  // Otherwise, it stores default initial value.
  DenseTensorMeta meta_;
  Place place_;
};

}  // namespace phi
