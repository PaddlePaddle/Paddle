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

#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_op.h"

namespace paddle {
namespace dialect {

pir::OpResult zeros_like(pir::OpResult x,
                         phi::DataType dtype,
                         const Place& place) {
  return paddle::dialect::full_like(x, 0, dtype, place);
}

pir::OpResult exponential__grad(pir::OpResult out_grad) {
  return paddle::dialect::zeros_like(out_grad);
}

pir::OpResult get_parameter(const std::string& name,
                            phi::DataType dtype,
                            const std::vector<int64_t>& shape) {
  phi::LoD lod;
  size_t offset{0};
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      TransToIrDataType(dtype),
      phi::DDim(shape.data(), shape.size()),
      phi::DataLayout::UNDEFINED,
      lod,
      offset);
  pir::GetParameterOp get_parameter_op =
      APIBuilder::Instance().GetBuilder()->Build<pir::GetParameterOp>(
          name, out_dense_tensor_type);
  return get_parameter_op.result(0);
}

void set_parameter(pir::OpResult parameter, const std::string& name) {
  APIBuilder::Instance().GetBuilder()->Build<pir::SetParameterOp>(parameter,
                                                                  name);
}

}  // namespace dialect
}  // namespace paddle
