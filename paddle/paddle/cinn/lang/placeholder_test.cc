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

#include "paddle/cinn/lang/placeholder.h"

#include <gtest/gtest.h>

#include "paddle/cinn/ir/ir_printer.h"

namespace cinn {
namespace lang {

TEST(placeholder, basic) {
  Expr M(100);
  Expr N(20);

  Placeholder<float> x("x", {M, N});

  ir::Var i("i");
  ir::Var j("j");

  auto slice = x(i, j);
  LOG(INFO) << "slice " << slice;
}

TEST(placeholder, dynamic_shape) {
  Var B("B", Int(32));
  Expr N(20);

  Placeholder<float> x("x", {B, N});

  Var i("i"), j("j");
  auto slice = x(i, j);
}

}  // namespace lang
}  // namespace cinn
