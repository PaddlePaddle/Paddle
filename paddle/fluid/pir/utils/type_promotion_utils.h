// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/phi/common/type_promotion.h"

namespace pir {

inline pir::Value PromoteCast(const std::string& input_name,
                              const pir::Value& input,
                              const phi::DataType& dst_dtype,
                              bool trace_backward = true) {
  if (paddle::imperative::GetDataType(input) != dst_dtype) {
    return paddle::imperative::Cast(input, dst_dtype, trace_backward);
  } else {
    return input;
  }
}

std::vector<int64_t> GetValueShape(const pir::Value& value) {
  if (value.type().isa<paddle::dialect::DenseTensorType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
  } else if (value.type().isa<paddle::dialect::SelectedRowsType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::SelectedRowsType>().dims());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get shape for dense_tensor or selected_rows."));
  }
}

}  // namespace pir
