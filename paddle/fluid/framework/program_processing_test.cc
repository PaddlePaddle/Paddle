/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/program_processing.h"

#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest.h"
#include "gtest/gtest_pred_impl.h"

namespace paddle {

namespace framework {

TEST(ProgramDesc, SSAprogram) {
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
  VLOG(3) << "sub_blocks_ID:" << sub_blocks[0]->ID();
  VLOG(3) << "sub_blocks_Parent:" << sub_blocks[0]->Parent();
  op->SetAttr("sub_blocks", sub_blocks);

  ProgramProcessor program_processor;
  program_processor.SSAProgram(&program);
}
}  // namespace framework
}  // namespace paddle
