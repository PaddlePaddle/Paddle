// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {
namespace ir_utils {

TEST(CollectIRNodes, basic0) {
  Expr C = Expr(1) + 2;

  auto exprs =
      CollectIRNodes(C, [](const Expr* x) { return x->As<ir::Add>(); });
  ASSERT_EQ(exprs.size(), 1UL);

  auto ints =
      CollectIRNodes(C, [](const Expr* x) { return x->As<ir::IntImm>(); });
  ASSERT_EQ(ints.size(), 2UL);
}

TEST(CollectIRNodes, basic) {
  Expr M(100);
  Expr N(200);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");

  auto stages = CreateStages({C});

  auto fn = Lower("fn", stages, {A, B, C});

  LOG(INFO) << "fn:\n" << fn;

  auto tensors =
      CollectIRNodes(fn, [](const Expr* x) { return x->as_tensor(); });
  ASSERT_EQ(tensors.size(), 5UL);

  auto fn_body = fn.As<ir::_LoweredFunc_>()->body;
  LOG(INFO) << "fn.body:\n" << fn_body;
  auto tensors2 =
      CollectIRNodes(fn_body, [](const Expr* x) { return x->as_tensor(); });
  auto exprs = CollectIRNodes(fn_body, [](const Expr* x) { return x; });
}
}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
