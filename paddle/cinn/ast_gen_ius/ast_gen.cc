// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ast_gen_ius/ast_gen.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn {
namespace ast_gen_ius {

ir::Expr AstGen::Build(const ir::Tensor& tensor) {
  const std::vector<ir::Var>& axis = tensor->axis();
  const std::vector<ir::Expr>& shape = tensor->shape;
  size_t axis_len = axis.size();
  CHECK_EQ(shape.size(), axis_len)
      << "Internal Error: Tensor has different shape and axis length in AstGen";
  ir::Expr body = tensor->body();
  for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
    ir::Var loop_var = axis[i];
    ir::Expr loop_extent = shape[i];
    body = ir::For::Make(loop_var,
                         Expr(0),
                         loop_extent,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         ir::Block::Make({body}));
  }
  return body;

  /*
  if (tensor->is_placeholder_node()) {
    return tensor;
  } else if (tensor->is_call_node()) {
    return tensor->operation->as<ir::CallOp>()->call_expr;
  } else if (tensor->is_compute_node()) {
    const ir::ComputeOp* compute_op = tensor->operation->as<ir::ComputeOp>();
    ir::Expr body = compute_op->body[0];
    size_t axis_len = compute_op->axis.size();
    CHECK_EQ(compute_op->shape.size(), axis_len)
        << "Internal Error: ComputeOp has different shape and axis";

    for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
      ir::Var loop_var = compute_op->axis[i];
      ir::Expr loop_extent = compute_op->shape[i];
      body = ir::For::Make(loop_var,
                           Expr(0),
                           loop_extent,
                           ir::ForType::Serial,
                           ir::DeviceAPI::Host,
                           ir::Block::Make({body}));
    }
    return body;
  } else {
    LOG(FATAL)
        << "Internal Error: unimplemented Tensor operation in AstGen::Build";
  }*/
}

}  // namespace ast_gen_ius
}  // namespace cinn
