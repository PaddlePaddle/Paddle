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

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/placeholder.h"

namespace cinn {
namespace ast_gen_ius {

using ir::Expr;
using ir::Tensor;
using ir::Var;
using lang::Compute;
using lang::Placeholder;

TEST(TensorGroup, Easy) {
  auto M = Expr(100);
  auto N = Expr(15);
  Placeholder<float> A("A", {M, N});

  Tensor B = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "B");

  TensorGroup tensor_group({B});

  ASSERT_TRUE(tensor_group.Contain("A"));
  ASSERT_TRUE(tensor_group.Contain("B"));
  ASSERT_EQ(tensor_group.Get("B")->name, "B");
  ASSERT_EQ(tensor_group.Get("A")->name, "A");
  ASSERT_EQ(tensor_group.GetAllTensors().size(), 2UL);

  ASSERT_EQ(tensor_group.GetCrtlDepTensors("A").size(), 0UL);
  ASSERT_EQ(tensor_group.GetCrtlDepTensors("B").size(), 1UL);
  ASSERT_TRUE(tensor_group.GetCrtlDepTensors("B").count(A));

  std::vector<ir::Tensor> topo_tensors =
      tensor_group.GetGenFuncTopoOrder({A.tensor(), B});
  ASSERT_EQ(topo_tensors.size(), 1UL);
  ASSERT_EQ(topo_tensors[0]->name, "B");
}

}  // namespace ast_gen_ius
}  // namespace cinn
