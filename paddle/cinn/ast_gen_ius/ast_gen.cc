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
  if (tensor->is_call_node()) {
    return tensor->operation->as<ir::CallOp>()->call_expr;
  } else if (tensor->is_compute_node()) {
    return tensor->operation->as<ir::ComputeOp>()->body[0];
  } else {
    LOG(FATAL)
        << "Internal Error: unimplemented Tensor operation in AstGen::Build";
  }
}

}  // namespace ast_gen_ius
}  // namespace cinn
