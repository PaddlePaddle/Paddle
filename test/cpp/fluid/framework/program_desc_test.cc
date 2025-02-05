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

#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest.h"
#include "gtest/gtest_pred_impl.h"

namespace paddle {
namespace framework {
class VarDesc;

TEST(ProgramDesc, block_desc_move) {
  auto program = std::make_unique<ProgramDesc>();
  auto* global_block = program->MutableBlock(0);

  auto* op = global_block->AppendOp();
  op->SetType("op_with_subblock");
  op->SetAttr("sub_block", program->AppendBlock(*global_block));

  std::vector<BlockDesc*> sub_blocks;
  sub_blocks.push_back(program->AppendBlock(*global_block));
  sub_blocks.push_back(program->AppendBlock(*global_block));
  op->SetAttr("sub_blocks", sub_blocks);

  program->Flush();

  ProgramDesc program_move;
  for (size_t i = 1; i < program->Size(); ++i) {
    program_move.AppendBlock(program_move.Block(0));
  }
  for (size_t i = 0; i < program->Size(); ++i) {
    program_move.MutableBlock(i)->MoveFrom(program->MutableBlock(i));
  }
  program = nullptr;
  EXPECT_EQ(program_move.Size(), static_cast<size_t>(4));
  op = program_move.Block(0).Op(0);
  auto sub_block = op->GetAttrIfExists<BlockDesc*>("sub_block");
  EXPECT_EQ(sub_block, program_move.MutableBlock(1));

  sub_blocks = op->GetAttrIfExists<std::vector<BlockDesc*>>("sub_blocks");
  EXPECT_EQ(sub_blocks.size(), static_cast<size_t>(2));
  EXPECT_EQ(sub_blocks[0], program_move.MutableBlock(2));
  EXPECT_EQ(sub_blocks[1], program_move.MutableBlock(3));
}

TEST(ProgramDesc, copy_ctor) {
  ProgramDesc program;
  auto* global_block = program.MutableBlock(0);
  auto* x = global_block->Var("X");
  x->SetType(proto::VarType::DENSE_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(proto::VarType::FP32);
  x->SetShape({1000, 784});

  auto* y = global_block->Var("Y");
  y->SetType(proto::VarType::DENSE_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(proto::VarType::FP32);
  y->SetShape({784, 100});

  auto* op = global_block->AppendOp();
  op->SetType("mul");
  op->SetInput("X", {x->Name()});
  op->SetInput("Y", {y->Name()});

  auto* out = global_block->Var("Out");
  out->SetType(proto::VarType::DENSE_TENSOR);
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
    auto op_origin = global_block->Op(static_cast<int>(i));
    auto op_copy = global_block_copy->Op(static_cast<int>(i));

    ASSERT_EQ(op_origin->Type(), op_copy->Type());
    ASSERT_EQ(op_origin->Inputs(), op_copy->Inputs());
    ASSERT_EQ(op_origin->Outputs(), op_copy->Outputs());

    ASSERT_EQ(op_origin->Proto()->attrs().size(),
              op_copy->Proto()->attrs().size());
    for (auto it = op_origin->Proto()->attrs().begin();
         it != op_origin->Proto()->attrs().end();
         ++it) {
      for (auto it_2 = op_copy->Proto()->attrs().begin();
           it_2 != op_copy->Proto()->attrs().end();
           ++it_2) {
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
  x->SetType(proto::VarType::DENSE_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(proto::VarType::FP32);
  x->SetShape({1000, 784});

  auto* y = global_block->Var("Y");
  y->SetType(proto::VarType::DENSE_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(proto::VarType::FP32);
  y->SetShape({784, 100});

  auto* op = global_block->AppendOp();
  op->SetType("mul");
  op->SetInput("X", {x->Name()});
  op->SetInput("Y", {y->Name()});

  auto* out = global_block->Var("Out");
  out->SetType(proto::VarType::DENSE_TENSOR);
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
    auto op_origin = global_block->Op(static_cast<int>(i));
    auto op_restored = global_block_restored->Op(static_cast<int>(i));

    ASSERT_EQ(op_origin->Type(), op_restored->Type());
    ASSERT_EQ(op_origin->Inputs(), op_restored->Inputs());
    ASSERT_EQ(op_origin->Outputs(), op_restored->Outputs());

    ASSERT_EQ(op_restored->Proto()->SerializeAsString(),
              op_origin->Proto()->SerializeAsString());
  }
}
}  // namespace framework
}  // namespace paddle
