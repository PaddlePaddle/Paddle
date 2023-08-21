// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void FullWithTensorKernel(
    const Context& dev_ctx,
    const paddle::optional<DenseTensor>& ValueTensor,
    const paddle::optional<DenseTensor>& ShapeTensor,
    // const paddle::optional<std::vector<DenseTensor>>& ShapeTensorList,
    const paddle::optional<std::vector<const DenseTensor*>>& ShapeTensorList,
    const std::vector<int64_t>& shape,
    float value,
    int dtype,
    const std::string& str_value,
    DenseTensor* out) {
  IntArray full_shape;
  Scalar full_val;
  DataType full_dtype;

  full_dtype = phi::VarTypeToDataType(
      static_cast<paddle::framework::proto::VarType_Type>(dtype));
  if (ValueTensor) {
    full_val = Scalar(*ValueTensor.get_ptr());
  } else if (!str_value.empty()) {
    full_val = Scalar(str_value);
  } else {
    full_val = Scalar(value);
  }

  if (ShapeTensor) {
    full_shape = IntArray(*ShapeTensor.get_ptr());
  } else if (ShapeTensorList) {
    // full_shape = IntArray(*ShapeTensorList.get_ptr());
  } else {
    full_shape = IntArray(shape);
  }

  FullKernel<T, Context>(dev_ctx, full_shape, full_val, full_dtype, out);
}
}  // namespace phi
