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

#include "paddle/cinn/ir/buffer.h"

#include <gtest/gtest.h>

#include <vector>

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/buffer.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"

namespace cinn {
namespace ir {

TEST(Buffer, basic) {
  Var ptr("buff", Float(32));
  std::vector<Expr> shape({Expr(100), Expr(20)});
  Var i("i"), j("j");
  std::vector<Expr> strides({Expr(0), Expr(0)});
  auto buffer = _Buffer_::Make(
      ptr, ptr->type(), shape, strides, Expr(0), "buf", "", 0, 0);

  // Check shared
  ASSERT_EQ(ref_count(buffer.get()).val(), 1);

  ASSERT_EQ(buffer->type(), type_of<cinn_buffer_t*>());
  ASSERT_EQ(buffer->dtype, ptr->type());

  {
    auto buffer1 = buffer;
    ASSERT_EQ(ref_count(buffer.get()).val(), 2);
    ASSERT_EQ(ref_count(buffer1.get()).val(), 2);
  }

  ASSERT_EQ(ref_count(buffer.get()).val(), 1);
}

TEST(Buffer, bind_to_multiple_tensors) {
  Expr M(100);
  Expr N(20);
  Tensor A = lang::Compute(
      {M, N}, [=](Var i, Var j) { return Expr(0.f); }, "A");
  Tensor B = lang::Compute(
      {M, N}, [=](Var i, Var j) { return Expr(1.f); }, "B");

  auto stages = CreateStages({A, B});

  stages[B]->ShareBufferWith(stages[A]);

  auto funcs = lang::Lower("func1", stages, {A, B});

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os = Target::OS ::Linux;

  ir::Module::Builder builder("module1", target);
  builder.AddFunction(funcs);
  builder.AddBuffer(A->buffer);

  backends::CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out =
      codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;
}

}  // namespace ir
}  // namespace cinn
