/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/program_desc.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/block_desc.h"

namespace paddle {
namespace framework {
TEST(ProgramDesc, copy_ctor) {
  ProgramDesc program;
  auto* global_block = program.MutableBlock(0);
  auto* x = global_block->Var("X");
  x->SetType(proto::VarType::LOD_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(proto::VarType::FP32);
  x->SetShape({1000, 784});

  auto* y = global_block->Var("Y");
  y->SetType(proto::VarType::LOD_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(proto::VarType::FP32);
  y->SetShape({784, 100});

  auto* op = global_block->AppendOp();
  op->SetType("mul");
  op->SetInput("X", {x->Name()});
  op->SetInput("Y", {y->Name()});

  auto* out = global_block->Var("Out");
  out->SetType(proto::VarType::LOD_TENSOR);
  op->SetOutput("Y", {out->Name()});

  BlockDesc* new_block = program.AppendBlock(*global_block);
  op = new_block->AppendOp();
  op->SetType("mul");

  op = global_block->AppendOp();
  op->SetType("op_with_subblock");
  op->SetAttr("sub_block", new_block);

  std::vector<BlockDesc*> sub_blocks;
  sub_blocks.push_back(program.AppendBlock(*global_block));
  sub_blocks.push_back(program.AppendBlock(*global_block));
  op->SetAttr("sub_blocks", sub_blocks);

  ProgramDesc program_copy(program);

  auto* global_block_copy = program_copy.MutableBlock(0);
  ASSERT_NE(global_block, global_block_copy);

  auto assert_same_var = [&](const std::string& name, VarDesc* var_before) {
    ASSERT_TRUE(global_block_copy->HasVar(name));
    auto* copy = global_block_copy->Var(name);
    ASSERT_NE(copy, var_before);
    ASSERT_EQ(copy->Name(), var_before->Name());
    ASSERT_EQ(copy->GetType(), var_before->GetType());
    ASSERT_EQ(copy->GetShape(), var_before->GetShape());
    ASSERT_EQ(copy->Proto()->SerializeAsString(),
              var_before->Proto()->SerializeAsString());
  };

  ASSERT_EQ(global_block->LocalVarNames(), global_block_copy->LocalVarNames());
  ASSERT_EQ(3UL, global_block_copy->LocalVarNames().size());
  assert_same_var("X", x);
  assert_same_var("Y", y);
  assert_same_var("Out", out);

  bool found_sub_block = false;
  bool found_sub_blocks = false;
  for (size_t i = 0; i < global_block->OpSize(); ++i) {
    auto op_origin = global_block->Op(i);
    auto op_copy = global_block_copy->Op(i);

    ASSERT_EQ(op_origin->Type(), op_copy->Type());
    ASSERT_EQ(op_origin->Inputs(), op_copy->Inputs());
    ASSERT_EQ(op_origin->Outputs(), op_copy->Outputs());

    ASSERT_EQ(op_origin->Proto()->attrs().size(),
              op_copy->Proto()->attrs().size());
    for (auto it = op_origin->Proto()->attrs().begin();
         it != op_origin->Proto()->attrs().end(); ++it) {
      for (auto it_2 = op_copy->Proto()->attrs().begin();
           it_2 != op_copy->Proto()->attrs().end(); ++it_2) {
        if (it->name() == it_2->name()) {
          ASSERT_TRUE(it_2->SerializeAsString() == it->SerializeAsString());
        }
      }
    }

    if (op->Type() == "op_with_subblock") {
      ASSERT_EQ(1, op->GetBlockAttrId("sub_block"));
      found_sub_block = true;

      ASSERT_EQ(2UL, op->GetBlocksAttrIds("sub_blocks").size());
      found_sub_blocks = true;
    }
  }
  ASSERT_TRUE(found_sub_block);
  ASSERT_TRUE(found_sub_blocks);
  // Not check block's protostr are same it because the order of vars could be
  // different and it is correct.
}

TEST(ProgramDescBind, serialize_and_deserialize) {
  ProgramDesc program_origin;
  auto* global_block = program_origin.MutableBlock(0);
  auto* x = global_block->Var("X");
  x->SetType(proto::VarType::LOD_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(proto::VarType::FP32);
  x->SetShape({1000, 784});

  auto* y = global_block->Var("Y");
  y->SetType(proto::VarType::LOD_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(proto::VarType::FP32);
  y->SetShape({784, 100});

  auto* op = global_block->AppendOp();
  op->SetType("mul");
  op->SetInput("X", {x->Name()});
  op->SetInput("Y", {y->Name()});

  auto* out = global_block->Var("Out");
  out->SetType(proto::VarType::LOD_TENSOR);
  op->SetOutput("Y", {out->Name()});

  std::string binary_str;
  program_origin.Proto()->SerializeToString(&binary_str);

  ProgramDesc program_restored(binary_str);
  auto* global_block_restored = program_restored.MutableBlock(0);
  ASSERT_NE(global_block, global_block_restored);

  auto assert_same_var = [&](const std::string& name, VarDesc* var_before) {
    ASSERT_TRUE(global_block_restored->HasVar(name));
    auto* restored = global_block_restored->Var(name);
    ASSERT_NE(restored, var_before);
    ASSERT_EQ(restored->Name(), var_before->Name());
    ASSERT_EQ(restored->GetType(), var_before->GetType());
    ASSERT_EQ(restored->GetShape(), var_before->GetShape());
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
    auto op_restored = global_block_restored->Op(i);

    ASSERT_EQ(op_origin->Type(), op_restored->Type());
    ASSERT_EQ(op_origin->Inputs(), op_restored->Inputs());
    ASSERT_EQ(op_origin->Outputs(), op_restored->Outputs());

    ASSERT_EQ(op_restored->Proto()->SerializeAsString(),
              op_origin->Proto()->SerializeAsString());
  }
}
}  // namespace framework
}  // namespace paddle
