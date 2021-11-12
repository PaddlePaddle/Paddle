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

#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {
/// \brief The VectorTensor is a special kind of Tensor, which has only
/// one dimension, and it's data is on the CPU.
/// Currently, we only allow int32 and int64 in the VectorTensor,
/// maybe it will support new data type in the future.
/// Besides, we can also take it's data on the other device in the future.
class VectorTensor : public TensorBase,
                     public TypeInfoTraits<TensorBase, VectorTensor> {
 public:
  // Constructor support implicit
  VectorTensor(const std::vector<int64_t>& vec);  // NOLINT

  // The dense_tensor must have one dim
  VectorTensor(const DenseTensor& dense_tensor);  // NOLINT

  // The DenseTensor in vec must have only one element
  VectorTensor(const std::vector<DenseTensor>& vec);  // NOLINT

  virtual ~VectorTensor();

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "VectorTensor"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const { return size_; }

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept {
    static DDim dim = paddle::framework::make_ddim({1});
    return dim;
  }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType data_type() const noexcept { return data_type_; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept { return DataLayout::ANY; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const {
    static Place place = paddle::platform::CPUPlace();
    return place;
  }

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept { return data_type_ != DataType::UNDEFINED; }

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  bool initialized() const { return data_type_ != DataType::UNDEFINED; }

  /// \brief Get the const data pointer value of type T.
  /// \return The const data pointer value of type T.
  template <typename T>
  const T* data() const;

 private:
  /// \brief Assign the data_ from const data pointer value of type T.
  template <typename T>
  void AssignData(const T* value_data) {
    if (value_data) {
      auto* data = static_cast<T*>(data_);
      for (auto i = 0; i < size_; i++) {
        data[i] = value_data[i];
      }
    } else {
      PADDLE_THROW(
          paddle::platform::errors::InvalidArgument("Null pointer error."));
    }
  }

 private:
  DataType data_type_ = DataType::UNDEFINED;
  int64_t size_ = 0;
  void* data_ = nullptr;
};

DDim GetDimFromVectorTensor(const VectorTensor& vector_tensor);

}  // namespace pten
