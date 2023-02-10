// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "paddle/fluid/framework/variable_utils.h"

#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace experimental {

phi::Scalar MakePhiScalarFromVar(const framework::Variable& variable) {
  auto expected_place = phi::TransToPhiPlace(phi::Backend::CPU);
  if (variable.IsType<phi::DenseTensor>()) {
    const auto& tensor = variable.Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        tensor.numel(),
        1UL,
        platform::errors::InvalidArgument("The DenseTensor used to construct "
                                          "the Scalar contains more than 1 "
                                          "value, it contains `%d` values.",
                                          tensor.numel()));
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      phi::DenseTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      return {tmp_tensor};
    } else {
      return {tensor};
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport casting input `%s` type to Scalar when call pt "
        "kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

phi::IntArray MakePhiIntArrayFromVar(const framework::Variable& variable) {
  if (variable.IsType<phi::DenseTensor>()) {
    const auto& tensor = variable.Get<phi::DenseTensor>();
    return MakePhiIntArray(tensor);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport casting input `%s` type to IntArray when call pt "
        "kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

// TODO(chentianyu03): Inplace with IntArray constructor
phi::IntArray MakePhiIntArrayFromVarList(
    const std::vector<framework::Variable*>& variable_list) {
  if (variable_list.size() == 0) {
    return phi::IntArray();
  }
  auto expected_place = phi::TransToPhiPlace(phi::Backend::CPU);

  std::vector<int64_t> vector_data;
  vector_data.reserve(variable_list.size());

  for (auto* var : variable_list) {
    paddle::experimental::DataType data_type;
    if (var->IsType<phi::DenseTensor>()) {
      const auto& tensor = var->Get<phi::DenseTensor>();
      data_type = tensor.dtype();
      if (data_type == paddle::experimental::DataType::INT64) {
        const auto& tensor = var->Get<phi::DenseTensor>();
        if (tensor.IsInitialized() &&
            !platform::is_same_place(tensor.place(), expected_place)) {
          phi::DenseTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int64_t>());
        } else {
          vector_data.push_back(*tensor.data<int64_t>());
        }
      } else if (data_type == paddle::experimental::DataType::INT32) {
        const auto& tensor = var->Get<phi::DenseTensor>();
        if (tensor.IsInitialized() &&
            !platform::is_same_place(tensor.place(), expected_place)) {
          phi::DenseTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int32_t>());
        } else {
          vector_data.push_back(*tensor.data<int32_t>());
        }
      } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Data type error. When cast a LoDTensor to VectorTensor, "
            "the data type of LoDTensor must be int32 or int64, "
            "but now data type is %s.",
            data_type));
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupport casting input `%s` type to VectorTensor when call pt "
          "kernel.",
          framework::ToTypeName(var->Type())));
    }
  }

  phi::IntArray result{vector_data};
  result.SetFromTensor(true);

  return result;
}

}  // namespace experimental
}  // namespace paddle
