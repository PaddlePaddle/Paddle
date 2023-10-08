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

#include <gtest/gtest.h>
#include <vector>

#include "paddle/cinn/ast_gen_ius/ast_gen.h"
#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/placeholder.h"

namespace cinn {
namespace ast_gen_ius {

using cinn::ir::Expr;
using cinn::ir::Tensor;

TEST(AstGen, Build) {
  std::vector<Expr> shape = {Expr(10), Expr(10), Expr(10), Expr(10)};
  lang::Placeholder<float> A("A", shape);
  Tensor B = lang::Compute(
      shape,
      [&](const std::vector<Expr>& indice) { return lang::Relu(A(indice), 0); },
      "relu_test");
  TensorGroup tensor_group({B});
  Expr out = AstGen::Build(B, &tensor_group);
  LOG(INFO) << out;
}

}  // namespace ast_gen_ius
}  // namespace cinn
