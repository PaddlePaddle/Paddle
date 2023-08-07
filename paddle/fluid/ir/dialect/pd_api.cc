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

#include "paddle/fluid/ir/dialect/pd_api.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/ir/core/builder.h"

namespace paddle {
namespace dialect {
ir::OpResult mean(ir::OpResult x, std::vector<int64_t> axis, bool keepdim) {
  paddle::dialect::MeanOp mean_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::MeanOp>(
          x, axis, keepdim);
  return mean_op.out();
}

ir::OpResult sum(ir::OpResult x,
                 std::vector<int64_t> axis,
                 phi::DataType dtype,
                 bool keepdim) {
  paddle::dialect::SumOp sum_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::SumOp>(
          x, axis, dtype, keepdim);
  return sum_op.out();
}

ir::OpResult divide(ir::OpResult x, ir::OpResult y) {
  paddle::dialect::DivideOp divide_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::DivideOp>(x,
                                                                            y);
  return divide_op.out();
}

ir::OpResult full(std::vector<int64_t> shape,
                  float value,
                  phi::DataType dtype,
                  phi::Place place) {
  paddle::dialect::FullOp full_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::FullOp>(
          shape, value, dtype, place);
  return full_op.out();
}

ir::OpResult tanh_grad(ir::OpResult out, ir::OpResult grad_out) {
  paddle::dialect::TanhGradOp tanh_grad_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::TanhGradOp>(
          out, grad_out);
  return tanh_grad_op.result(0);
}

ir::OpResult mean_grad(ir::OpResult x,
                       ir::OpResult out_grad,
                       std::vector<int64_t> axis,
                       bool keepdim,
                       bool reduce_all) {
  paddle::dialect::MeanGradOp mean_grad_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::MeanGradOp>(
          x, out_grad, axis, keepdim, reduce_all);
  return mean_grad_op.result(0);
}

}  // namespace dialect
}  // namespace paddle
