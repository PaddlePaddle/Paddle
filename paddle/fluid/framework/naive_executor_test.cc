// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/naive_executor.h"
#include <gtest/gtest.h>
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

TEST(NaiveExecutor, Basic) {
  ProgramDesc program;
  auto* main_block = program.MutableBlock(0);
  auto* a = main_block->Var("a");  // input
  auto* b = main_block->Var("b");  // input
  auto* c = main_block->Var("c");  // input
  a->SetType(proto::VarType::LOD_TENSOR);
  b->SetType(proto::VarType::LOD_TENSOR);
  c->SetType(proto::VarType::LOD_TENSOR);

  auto* add = main_block->AppendOp();
  add->SetType("elementwise_add");
  add->SetInput("X", {"a"});
  add->SetInput("Y", {"b"});
  add->SetOutput("Out", {"c"});

  auto place = platform::CPUPlace();
  NaiveExecutor exe(place);
  exe.Prepare(nullptr, program, 0, false);
  auto* a_tensor = exe.FindTensor("a");
  auto* b_tensor = exe.FindTensor("b");
  auto* c_tensor = exe.FindTensor("c");

  a_tensor->Resize({1, 4});
  b_tensor->Resize({1, 4});
  c_tensor->Resize({1, 4});
  b_tensor->mutable_data<float>(place);
  a_tensor->mutable_data<float>(place);

  float a_arr[] = {0, 1, 2, 3};
  float b_arr[] = {0.0, .1, .2, .3};

  std::copy_n(a_arr, 4, a_tensor->mutable_data<float>(place));
  std::copy_n(b_arr, 4, b_tensor->mutable_data<float>(place));

  exe.Run();

  auto* c_data = c_tensor->mutable_data<float>(place);
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(c_data[i], 1.1 * i, 1e-3);
  }
}

}  // namespace framework
}  // namespace paddle

USE_OP_ITSELF(elementwise_add);
