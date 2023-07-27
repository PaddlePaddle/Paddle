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

#include "paddle/fluid/ir/dialect/ir_api.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"

namespace ir {
namespace api {
std::vector<ir::OpResult> tanh_grad(ir::OpResult out, ir::OpResult grad_out) {
  std::vector<ir::OpResult> res;
  // 1.get insert block
  ir::Block* insert_block_ptr = grad_out.owner()->GetParent();
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  // 2. construct builder
  ir::Builder builder = ir::Builder(ctx, insert_block_ptr);

  // 3. build op
  paddle::dialect::TanhGradOp grad_op =
      builder.Build<paddle::dialect::TanhGradOp>(out, grad_out);
  // 4. get op's output
  res.push_back(grad_op.x_grad());
  return res;
}
}  // namespace api
}  // namespace ir
