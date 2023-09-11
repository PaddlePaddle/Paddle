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
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_api.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_op.h"

namespace paddle {
namespace dialect {
pir::OpResult split_grad(std::vector<pir::OpResult> out_grads,
                         pir::OpResult axis) {
  auto combine_op =
      APIBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grads);
  paddle::dialect::SplitGradOp split_grad_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          combine_op.out(), axis);

  return split_grad_op.x_grad();
}

pir::OpResult split_grad(std::vector<pir::OpResult> out_grads, int axis) {
  auto combine_op =
      APIBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grads);
  paddle::dialect::SplitGradOp split_grad_op =
      APIBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          combine_op.out(), axis);

  return split_grad_op.x_grad();
}

pir::OpResult embedding_grad(pir::OpResult x,
                             pir::OpResult weight,
                             pir::OpResult out_grad,
                             int64_t padding_idx,
                             bool sparse) {
  if (weight.type().isa<paddle::dialect::DenseTensorType>()) {
    if (sparse) {
      return paddle::dialect::embedding_grad_sparse(
          x, weight, out_grad, padding_idx, sparse);
    } else {
      return paddle::dialect::embedding_grad_dense(
          x, weight, out_grad, padding_idx, sparse);
    }
  } else {
    if (sparse) {
      return paddle::dialect::sparse_weight_embedding_grad_sparse(
          x, weight, out_grad, padding_idx, sparse);
    } else {
      return paddle::dialect::sparse_weight_embedding_grad_dense(
          x, weight, out_grad, padding_idx, sparse);
    }
  }
}

}  // namespace dialect
}  // namespace paddle
