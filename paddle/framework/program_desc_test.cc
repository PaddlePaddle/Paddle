/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/program_desc.h"
#include "gtest/gtest.h"
#include "paddle/framework/block_desc.h"

namespace paddle {
namespace framework {
TEST(ProgramDesc, copy_ctor) {
  ProgramDescBind program;
  auto* global_block = program.Block(0);
  auto* x = global_block->Var("X");
  x->SetType(VarDesc_VarType_LOD_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(FP32);
  x->SetShape({1000, 784});

  auto* y = global_block->Var("Y");
  y->SetType(VarDesc_VarType_LOD_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(FP32);
  y->SetShape({784, 100});

  auto* op = global_block->AppendOp();
  op->SetType("mul");
  op->SetInput("X", {x->Name()});
  op->SetInput("Y", {y->Name()});

  auto* out = global_block->Var("Out");
  out->SetType(VarDesc_VarType_LOD_TENSOR);
  op->SetOutput("Y", {out->Name()});

  ProgramDescBind program_copy(program);

  auto* global_block_copy = program_copy.Block(0);
  ASSERT_NE(global_block, global_block_copy);

  auto assert_same_var = [&](const std::string& name, VarDescBind* var_before) {
    ASSERT_TRUE(global_block_copy->HasVar(name));
    auto* copy = global_block_copy->Var(name);
    ASSERT_NE(copy, var_before);
    ASSERT_EQ(copy->Name(), var_before->Name());
    ASSERT_EQ(copy->GetType(), var_before->GetType());
    ASSERT_EQ(copy->Shape(), var_before->Shape());
    ASSERT_EQ(copy->Proto()->SerializeAsString(),
              var_before->Proto()->SerializeAsString());
  };

  ASSERT_EQ(global_block->LocalVarNames(), global_block_copy->LocalVarNames());
  ASSERT_EQ(3, global_block_copy->LocalVarNames().size());
  assert_same_var("X", x);
  assert_same_var("Y", y);
  assert_same_var("Out", out);

  for (size_t i = 0; i < global_block->NumOfOps(); ++i) {
    auto op_origin = global_block->MutableOp(i);
    auto op_copy = global_block->MutableOp(i);
    ASSERT_EQ(op_copy->Proto()->SerializeAsString(),
              op_origin->Proto()->SerializeAsString());
  }

  // Not check block's protostr are same it because the order of vars could be
  // different and it is correct.
}
}  // namespace framework
}  // namespace paddle