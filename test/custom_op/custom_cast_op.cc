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

#include <iostream>
#include <vector>

#include "paddle/extension.h"

paddle::DataType ConvertDtype(const std::string& data_type) {
  if (data_type == "float16") {
    return paddle::DataType::FLOAT16;
  } else if (data_type == "float32") {
    return paddle::DataType::FLOAT32;
  } else if (data_type == "float64") {
    return paddle::DataType::FLOAT64;
  } else {
    PD_THROW("DataType Not Supported.");
  }
}

std::vector<paddle::Tensor> CastForward(const paddle::Tensor& x,
                                        const std::string& data_type) {
  return {paddle::experimental::cast(x, ConvertDtype(data_type))};
}

std::vector<paddle::DataType> CastForwardInferDtype(
    const paddle::DataType& input_dtype, const std::string& data_type) {
  return {ConvertDtype(data_type)};
}

std::vector<paddle::Tensor> CastBackward(const paddle::Tensor& grad_out,
                                         const std::string& data_type) {
  return {paddle::experimental::cast(grad_out, ConvertDtype(data_type))};
}

std::vector<paddle::DataType> CastBackwardInferDtype(
    const paddle::DataType& grad_out_dtype, const std::string& data_type) {
  return {ConvertDtype(data_type)};
}

PD_BUILD_OP(custom_cast)
    .Inputs({"X"})
    .Attrs({"data_type: std::string"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CastForward))
    .SetInferDtypeFn(PD_INFER_DTYPE(CastForwardInferDtype));

PD_BUILD_GRAD_OP(custom_cast)
    .Inputs({paddle::Grad("Out")})
    .Attrs({"data_type: std::string"})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(CastBackward))
    .SetInferDtypeFn(PD_INFER_DTYPE(CastBackwardInferDtype));
