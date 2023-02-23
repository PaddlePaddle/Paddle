//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;

inline std::string GetValueName(framework::proto::VarType::Type data_type) {
  std::string value_name;
  switch (data_type) {
    case framework::proto::VarType::INT32:
      value_name = "int32_values";
      break;
    case framework::proto::VarType::INT64:
      value_name = "int64_values";
      break;
    case framework::proto::VarType::FP32:
      value_name = "fp32_values";
      break;
    case framework::proto::VarType::FP64:
      value_name = "fp64_values";
      break;
    case framework::proto::VarType::BOOL:
      value_name = "bool_values";
      break;

    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type(code %d) for SetValue operator, only "
          "supports bool, int32, float32 and int64.",
          data_type));
  }
  return value_name;
}

// check whether the tensor with dimension of second can assign to the
// tensor with dimension of first
inline void CheckIsDimsMatch(const framework::DDim first,
                             const framework::DDim second) {
  int ignore_axis1 = 0, ignore_axis2 = 0;
  for (; ignore_axis1 < first.size(); ++ignore_axis1) {
    if (first[ignore_axis1] != 1) {
      break;
    }
  }
  for (; ignore_axis2 < second.size(); ++ignore_axis2) {
    if (second[ignore_axis2] != 1) {
      break;
    }
  }

  if (second.size() == ignore_axis2) {
    // second tensor has only one value
    return;
  }

  if (first.size() - ignore_axis1 >= second.size() - ignore_axis2) {
    auto idx1 = first.size() - 1;
    auto idx2 = second.size() - 1;
    bool is_match = true;
    for (; idx2 >= ignore_axis2; idx2--) {
      if (first[idx1--] != second[idx2] && second[idx2] != 1) {
        is_match = false;
        break;
      }
    }
    if (is_match) {
      return;
    }
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "The shape of tensor assigned value must match the shape "
      "of target shape: %d, but now shape is %d.",
      second.to_str(),
      first.to_str()));
}

}  // namespace operators
}  // namespace paddle
