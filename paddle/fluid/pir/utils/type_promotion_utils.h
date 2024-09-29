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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/common/type_promotion.h"

namespace pir {

inline pir::Value PromoteCast(const std::string& input_name,
                              const pir::Value& input,
                              const phi::DataType& dst_dtype) {
  if (paddle::dialect::GetValueDataType(input) != dst_dtype) {
    return paddle::dialect::cast(input, dst_dtype);
  } else {
    return input;
  }
}

inline void PromoteCastInplace(const std::string& input_name,
                               const pir::Value& input,
                               const phi::DataType& dst_dtype) {
  if (paddle::dialect::GetValueDataType(input) != dst_dtype) {
    paddle::dialect::cast_(input, dst_dtype);
  }
}

std::vector<int64_t> GetValueShape(const pir::Value& value) {
  if (value.type().isa<paddle::dialect::DenseTensorType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
  } else if (value.type().isa<paddle::dialect::DistDenseTensorType>()) {
    return phi::vectorize(value.type()
                              .dyn_cast<paddle::dialect::DistDenseTensorType>()
                              .global_ddim());
  } else if (value.type().isa<paddle::dialect::SelectedRowsType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::SelectedRowsType>().dims());
  } else if (value.type().isa<paddle::dialect::SparseCooTensorType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::SparseCooTensorType>().dims());
  } else if (value.type().isa<paddle::dialect::SparseCsrTensorType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::SparseCsrTensorType>().dims());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get shape for dense_tensor, selected_rows, "
        "sparse_coo_tensor, sparse_csr_tensor and dist_dense_tensor."));
  }
}

}  // namespace pir
