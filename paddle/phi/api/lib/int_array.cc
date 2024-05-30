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

#include "paddle/phi/common/int_array.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/tensor_copy.h"
#include "paddle/phi/common/place.h"

namespace paddle::experimental {

template <>
IntArrayBase<Tensor>::IntArrayBase(const Tensor& tensor) {  // NOLINT
  is_from_tensor_ = true;
  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    AssignDataFromTensor(tensor);
  } else {
    Tensor tensor_tmp;
    copy(tensor, phi::CPUPlace(), true, &tensor_tmp);
    AssignDataFromTensor(tensor_tmp);
  }
}

template <>
IntArrayBase<Tensor>::IntArrayBase(const std::vector<Tensor>& tensor_list) {
  is_from_tensor_ = true;

  for (const auto& tensor : tensor_list) {
    DataType data_type = tensor.dtype();
    switch (data_type) {
      case DataType::INT32:
        if (tensor.place().GetType() == AllocationType::CPU) {
          array_.push_back(*tensor.template data<int32_t>());
        } else {
          Tensor tensor_tmp;
          copy(tensor, phi::CPUPlace(), true, &tensor_tmp);
          array_.push_back(*tensor_tmp.template data<int32_t>());
        }
        break;
      case DataType::INT64:
        if (tensor.place().GetType() == AllocationType::CPU) {
          array_.push_back(*tensor.template data<int64_t>());
        } else {
          Tensor tensor_tmp;
          copy(tensor, phi::CPUPlace(), true, &tensor_tmp);
          array_.push_back(*tensor_tmp.template data<int64_t>());
        }
        break;
      default:
        PD_THROW(
            "Data type error. Currently, The data type of IntArrayBase "
            "only supports Tensor with int32 and int64, "
            "but now received `",
            data_type,
            "`.");
    }
  }
}

}  // namespace paddle::experimental
