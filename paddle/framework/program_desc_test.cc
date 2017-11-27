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
  auto* global_block = program.MutableBlock(0);
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

  auto* global_block_copy = program_copy.MutableBlock(0);
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
  ASSERT_EQ(3UL, global_block_copy->LocalVarNames().size());
  assert_same_var("X", x);
  assert_same_var("Y", y);
  assert_same_var("Out", out);

  for (size_t i = 0; i < global_block->OpSize(); ++i) {
    auto op_origin = global_block->Op(i);
    auto op_copy = global_block->Op(i);

    ASSERT_EQ(op_origin->Type(), op_copy->Type());
    ASSERT_EQ(op_origin->Inputs(), op_copy->Inputs());
    ASSERT_EQ(op_origin->Outputs(), op_copy->Outputs());

    ASSERT_EQ(op_copy->Proto()->SerializeAsString(),
              op_origin->Proto()->SerializeAsString());
  }

  // Not check block's protostr are same it because the order of vars could be
  // different and it is correct.
}

TEST(ProgramDescBind, serialize_and_deserialize) {
  ProgramDescBind program_origin;
  auto* global_block = program_origin.MutableBlock(0);
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

  std::string binary_str;
  program_origin.Proto()->SerializeToString(&binary_str);

  ProgramDescBind program_restored(binary_str);
  auto* global_block_restored = program_restored.MutableBlock(0);
  ASSERT_NE(global_block, global_block_restored);

  auto assert_same_var = [&](const std::string& name, VarDescBind* var_before) {
    ASSERT_TRUE(global_block_restored->HasVar(name));
    auto* restored = global_block_restored->Var(name);
    ASSERT_NE(restored, var_before);
    ASSERT_EQ(restored->Name(), var_before->Name());
    ASSERT_EQ(restored->GetType(), var_before->GetType());
    ASSERT_EQ(restored->Shape(), var_before->Shape());
    ASSERT_EQ(restored->Proto()->SerializeAsString(),
              var_before->Proto()->SerializeAsString());
  };

  ASSERT_EQ(global_block->LocalVarNames(),
            global_block_restored->LocalVarNames());
  ASSERT_EQ(3UL, global_block_restored->LocalVarNames().size());
  assert_same_var("X", x);
  assert_same_var("Y", y);
  assert_same_var("Out", out);

  for (size_t i = 0; i < global_block->OpSize(); ++i) {
    auto op_origin = global_block->Op(i);
    auto op_restored = global_block->Op(i);

    ASSERT_EQ(op_origin->Type(), op_restored->Type());
    ASSERT_EQ(op_origin->Inputs(), op_restored->Inputs());
    ASSERT_EQ(op_origin->Outputs(), op_restored->Outputs());

    ASSERT_EQ(op_restored->Proto()->SerializeAsString(),
              op_origin->Proto()->SerializeAsString());
  }
}
}  // namespace framework
}  // namespace paddle
