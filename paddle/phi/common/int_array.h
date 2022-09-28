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

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace experimental {

template <typename T>
class IntArrayBase {
 public:
  // Constructor support implicit
  IntArrayBase() = default;

  IntArrayBase(const std::vector<int64_t>& vec) : array_(vec) {}  // NOLINT

  IntArrayBase(const std::vector<int32_t>& vec) {  // NOLINT
    array_.insert(array_.begin(), vec.begin(), vec.end());
  }

  IntArrayBase(std::initializer_list<int64_t> array_list)
      : array_(array_list) {}

  IntArrayBase(const int64_t* data_value, int64_t n) {
    AssignData(data_value, n);
  }

  IntArrayBase(const int32_t* data_value, int64_t n) {
    AssignData(data_value, n);
  }

  bool FromTensor() const { return is_from_tensor_; }

  void SetFromTensor(bool val) { is_from_tensor_ = val; }

  // The Tensor must have one dim
  IntArrayBase(const T& tensor);  // NOLINT

  // The Tensor in vec must have only one element
  IntArrayBase(const std::vector<T>& tensor_list);  // NOLINT

  template <typename OtherT>
  IntArrayBase(const IntArrayBase<OtherT>& other) : array_(other.GetData()) {}

  size_t size() const { return array_.size(); }

  const std::vector<int64_t>& GetData() const { return array_; }

 private:
  /// \brief Assign the data_ from const data pointer value of type T.
  template <typename TYPE>
  void AssignData(const TYPE* value_data, int64_t n) {
    if (value_data || n == 0) {
      array_.reserve(n);
      for (auto i = 0; i < n; ++i) {
        array_.push_back(static_cast<int64_t>(value_data[i]));
      }
    } else {
      PD_THROW("The input data pointer is null.");
    }
  }

  void AssignDataFromTensor(const T& tensor) {
    size_t n = tensor.numel();
    array_.reserve(n);
    switch (tensor.dtype()) {
      case DataType::INT32:
        AssignData(tensor.template data<int32_t>(), n);
        break;
      case DataType::INT64:
        AssignData(tensor.template data<int64_t>(), n);
        break;
      default:
        PD_THROW(
            "Data type error. Currently, The data type of IntArrayBase "
            "only supports Tensor with int32 and int64, "
            "but now received `",
            tensor.dtype(),
            "`.");
    }
  }

 private:
  // TODO(zhangyunfei) Replace std::vector with a more efficient container
  // structure.
  std::vector<int64_t> array_;
  bool is_from_tensor_{false};
};

using IntArray =
    paddle::experimental::IntArrayBase<paddle::experimental::Tensor>;

}  // namespace experimental
}  // namespace paddle

namespace phi {

class DenseTensor;
using IntArray = paddle::experimental::IntArrayBase<DenseTensor>;

}  // namespace phi
