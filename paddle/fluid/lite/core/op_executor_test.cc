// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/op_executor.h"
#include <gtest/gtest.h>
#include <vector>

namespace paddle {
namespace lite {

TEST(executor, test) {
  std::vector<Place> valid_places{Place{TARGET(kHost), PRECISION(kFloat)}};

  auto scope = std::make_shared<lite::Scope>();

  framework::ProgramDesc program;
  program.MutableBlock(0)->Var("x");
  program.MutableBlock(0)->Var("bias")->SetPersistable(true);
  program.MutableBlock(0)->Var("w")->SetPersistable(true);
  program.MutableBlock(0)->Var("output");

  auto& op_desc = *program.MutableBlock(0)->AppendOp();
  op_desc.SetType("fc");
  op_desc.SetInput("Input", {"x"});
  op_desc.SetInput("W", {"w"});
  op_desc.SetInput("Bias", {"bias"});
  op_desc.SetOutput("Out", {"output"});
  op_desc.SetAttr("in_num_col_dims", static_cast<int>(1));
  program.Flush();

  auto* w = scope->Var("w")->GetMutable<Tensor>();
  w->Resize({20, 20});
  auto* x = scope->Var("x")->GetMutable<Tensor>();
  x->Resize({1, 10, 20});
  auto* bias = scope->Var("bias")->GetMutable<Tensor>();
  bias->Resize({1, 20});

  bias->mutable_data<float>();
  w->mutable_data<float>();
  x->mutable_data<float>();

  lite::Executor executor(program, scope, valid_places);
  executor.Run();
}

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_LITE_KERNEL(fc, kHost, kFloat, def);
