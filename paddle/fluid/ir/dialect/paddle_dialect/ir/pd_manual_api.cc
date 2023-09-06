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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_api.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/api_builder.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/ir/core/builtin_op.h"

namespace paddle {
namespace dialect {
ir::OpResult split_grad(std::vector<ir::OpResult> out_grads,
                        ir::OpResult axis) {
  auto combine_op =
      APIBuilder::Instance().GetBuilder()->Build<ir::CombineOp>(out_grads);
  paddle::dialect::SplitGradOp split_grad_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          combine_op.out(), axis);

  return split_grad_op.x_grad();
}

ir::OpResult split_grad(std::vector<ir::OpResult> out_grads, int axis) {
  auto combine_op =
      APIBuilder::Instance().GetBuilder()->Build<ir::CombineOp>(out_grads);
  paddle::dialect::SplitGradOp split_grad_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          combine_op.out(), axis);

  return split_grad_op.x_grad();
}
ir::OpResult get_parameter(const std::string& name,
                           phi::DataType dtype,
                           const std::vector<int64_t>& shape) {
  phi::LoD lod;
  size_t offset{0};
  ir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      ir::IrContext::Instance(),
      TransToIrDataType(dtype),
      phi::DDim(shape.data(), shape.size()),
      phi::DataLayout::UNDEFINED,
      lod,
      offset);
  ir::GetParameterOp get_parameter_op =
      APIBuilder::Instance().GetBuilder()->Build<ir::GetParameterOp>(
          name, out_dense_tensor_type);
  return get_parameter_op.result(0);
}

void set_parameter(ir::OpResult parameter, const std::string& name) {
  APIBuilder::Instance().GetBuilder()->Build<ir::SetParameterOp>(parameter,
                                                                 name);
}

}  // namespace dialect
}  // namespace paddle
