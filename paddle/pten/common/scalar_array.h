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

#include "paddle/pten/api/ext/exception.h"
#include "paddle/pten/api/include/tensor.h"

namespace paddle {
namespace experimental {

template <typename T>
class ScalarArrayBase {
 public:
  // Constructor support implicit
  ScalarArrayBase() = default;

  ScalarArrayBase(const std::vector<int64_t>& vec) : array_(vec) {}  // NOLINT

  ScalarArrayBase(const int64_t* date_value, int64_t n) {
    AssignData(date_value, n);
  }

  ScalarArrayBase(const int32_t* date_value, int64_t n) {
    AssignData(date_value, n);
  }

  // The Tensor must have one dim
  ScalarArrayBase(const T& tensor) {  // NOLINT
    size_t n = tensor.numel();
    array_.reserve(n);
    switch (tensor.type()) {
      case DataType::INT32:
        AssignData(tensor.template data<int32_t>(), n);
        break;
      case DataType::INT64:
        AssignData(tensor.template data<int64_t>(), n);
        break;
      default:
        PD_THROW(
            "Data type error. Currently, The data type of ScalarArrayBase "
            "only supports Tensor with int32 and int64, "
            "but now received `",
            tensor.type(),
            "`.");
    }
  }

  // The Tensor in vec must have only one element
  ScalarArrayBase(const std::vector<T>& tensor_list) {  // NOLINT
    auto n = tensor_list.size();
    array_.reserve(n);
    if (!tensor_list.empty()) {
      DataType data_type = tensor_list[0].dtype();
      switch (data_type) {
        case DataType::INT32: {
          for (size_t i = 0; i < n; i++) {
            PD_CHECK(tensor_list[i].dtype() == data_type,
                     "The data_type of tensors in the list isn't consistent."
                     "the first tensor is`",
                     data_type,
                     "` but `",
                     i,
                     "`th tensor is`",
                     tensor_list[i].dtype(),
                     "`.");
            array_.push_back(*tensor_list[i].template data<int32_t>());
          }
          break;
        }
        case DataType::INT64: {
          for (size_t i = 0; i < n; i++) {
            PD_CHECK(tensor_list[i].dtype() == data_type,
                     "The data_type of tensors in the list isn't consistent."
                     "the first tensor is`",
                     data_type,
                     "` but `",
                     i,
                     "`th tensor is`",
                     tensor_list[i].dtype(),
                     "`.");
            array_.push_back(*tensor_list[i].template data<int64_t>());
          }
          break;
        }
        default:
          PD_THROW(
              "Data type error. Currently, The data type of ScalarArrayBase "
              "only supports Tensor with int32 and int64, "
              "but now received `",
              data_type,
              "`.");
      }
    }
  }

  template <typename TT>
  ScalarArrayBase(const ScalarArrayBase<TT>& other) : array_(other.GetData()) {}

  const std::vector<int64_t>& GetData() const { return array_; }

 private:
  /// \brief Assign the data_ from const data pointer value of type T.
  template <typename TYPE>
  void AssignData(const TYPE* value_data, int64_t n) {
    if (value_data) {
      array_.reserve(n);
      for (auto i = 0; i < n; i++) {
        array_.push_back(static_cast<int64_t>(value_data[i]));
      }
    } else {
      PD_THROW("The input data pointer is null.");
    }
  }

  // template <typename TT>
  // friend paddle::framework::DDim GetDimFromScalarArray(
  //     const ScalarArrayBase<TT>& scalar_array);

 private:
  std::vector<int64_t> array_;
};

// template <typename T>
// paddle::framework::DDim GetDimFromScalarArray(
//     const ScalarArrayBase<T>& scalar_array) {
//   return paddle::framework::make_ddim(scalar_array.array_);
// }

using ScalarArray =
    paddle::experimental::ScalarArrayBase<paddle::experimental::Tensor>;

}  // namespace experimental
}  // namespace paddle

namespace pten {

class DenseTensor;
using ScalarArray = paddle::experimental::ScalarArrayBase<DenseTensor>;
}
