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

#include <unordered_map>

#include "paddle/phi/core/extended_tensor.h"
#include "paddle/utils/any.h"

namespace paddle {
namespace framework {

/// \brief Fluid Kernel and PHI Kernel will be unified in the future.
/// So, we need a class in PHI that can represent the RAW type in Fluid.
/// The RawTensor is for PHI Kernel that has RAW type arguments.
class RawTensor : public phi::ExtendedTensor,
                  public phi::TypeInfoTraits<phi::TensorBase, RawTensor> {
 public:
  RawTensor() = default;

  RawTensor(RawTensor&& other) = default;

  RawTensor(const RawTensor& other) = default;

  RawTensor& operator=(RawTensor&& other) = default;

  /// \brief Destroy the RawTensor and release exclusive resources.
  virtual ~RawTensor() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "RawTensor"; }

  template <typename T>
  T* GetMutable() {
    if (!data_.empty()) {
      try {
        return paddle::any_cast<T*>(data_);
      } catch (paddle::bad_any_cast&) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Invalid data type error, expected %s, actual %s.",
            typeid(T).name(),
            data_type_.name()));
      }
    }
    T* created_data = new T();
    data_ = created_data;
    data_deleter_ = [created_data]() { delete created_data; };
    data_type_ = std::type_index(typeid(T));
    return created_data;
  }

  template <typename T>
  bool IsType() const {
    return std::type_index(typeid(T)) == data_type_;
  }

 private:
  paddle::any data_;
  std::function<void(void)> data_deleter_;
  std::type_index data_type_ = std::type_index(typeid(void));
};

}  // namespace framework
}  // namespace paddle
