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
#include "paddle/cinn/common/target.h"

namespace cinn {

Expr batch(256);
Expr in_channel(256);
Expr out_channel(512);
Expr in_size(14);
Expr kernel(3);
Expr pad(1);
Expr stride(1);

TEST(test03_conv, basic) {
  Placeholder<float> A("A", {in_size, in_size, in_channel, batch});
  Placeholder<float> W("W", {kernel, kernel, in_channel, out_channel});
  Expr out_size = (in_size - kernel + 2 * pad) / stride + 1;

  auto Apad = Compute(
      {in_size + 2 * pad, in_size + 2 * pad, in_channel, batch},
      [&](Expr yy, Expr xx, Expr cc, Expr nn) {
        auto cond = logic_and(
            {yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size});
        return ir::Select::Make(cond, A(yy - pad, xx - pad, cc, nn), Expr(0.f));
      },
      "Apad");

  Var rc(Expr(0), Expr(in_channel), "rc");
  Var ry(Expr(0), Expr(kernel), "ry");
  Var rx(Expr(0), Expr(kernel), "rx");

  auto B = Compute(
      {out_size, out_size, out_channel, batch},
      [&](Expr yy, Expr xx, Expr ff, Expr nn) {
        return ReduceSum(Apad(yy * stride + ry, xx * stride + rx, rc, nn) *
                             W(ry, rx, rc, ff),
                         {rx, ry, rc});
      },
      "B");

  Target target = common::DefaultHostTarget();

  Module::Builder builder("conv", target);

  auto stages = CreateStages({Apad, B});

  auto func = Lower("conv", stages, {A, W, Apad, B});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test03_convolution.h")
                .c_source("./test03_convolution.cc");
  compiler.Compile(builder.Build(), outputs);
}

}  // namespace cinn
