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

  auto* out1 = global_block->Var("Out");
  out1->SetType(proto::VarType::LOD_TENSOR);
  op->SetOutput("Y", {out1->Name()});

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

  // building cond op such as less_than
  BlockDesc* parent_block = program.MutableBlock(sub_blocks[0]->Parent());
  op = parent_block->AppendOp();
  op->SetType("less_than");
  auto* x1 = parent_block->Var("X");
  x1->SetType(proto::VarType::LOD_TENSOR);
  x1->SetLoDLevel(0);
  x1->SetDataType(proto::VarType::FP32);
  x1->SetShape({1});

  auto* y1 = parent_block->Var("Y");
  y1->SetType(proto::VarType::LOD_TENSOR);
  y1->SetLoDLevel(0);
  y1->SetDataType(proto::VarType::FP32);
  y1->SetShape({1});

  op->SetInput("X", {x1->Name()});
  op->SetInput("Y", {y1->Name()});

  auto* less_than_out = parent_block->Var("Out");
  out1->SetType(proto::VarType::BOOL);
  op->SetOutput("Out", {less_than_out->Name()});

  // building while op
  // BlockDesc* parent_block = program.MutableBlock(sub_blocks[0]->Parent());
  op = sub_blocks[0]->AppendOp();
  op->SetType("while");
  auto* x2 = parent_block->Var("X1");

  x2->SetType(proto::VarType::LOD_TENSOR);
  x2->SetLoDLevel(0);
  x2->SetDataType(proto::VarType::FP32);
  x2->SetShape({1});

  // auto* Condition = parent_block->Var("Condition");
  // Condition->SetType(proto::VarType::BOOL);

  op->SetInput("kX", {x2->Name()});
  op->SetInput("kCondition", {less_than_out->Name()});

  auto* out = sub_blocks[0]->Var("Out");
  out->SetType(proto::VarType::LOD_TENSOR);
  out->SetLoDLevel(0);
  out->SetDataType(proto::VarType::FP32);
  out->SetShape({1});

  auto* steps = sub_blocks[0]->Var("StepScopes");
  // steps->SetType(proto::VarType::STEP_SCOPES);
  // steps->SetDataType(proto::VarType::FP32);
  // steps->SetShape({1});

  op->SetOutput("kOutputs", {out->Name()});
  op->SetOutput("kStepScopes", {steps->Name()});

  ProgramProcessor program_processor;
  // program_processor.SSAProgram(&program);

  std::set<std::string> x_name_list;
  std::set<std::string> inner_outputs;

  program_processor.GetInputsOutputsInBlock(&program, *sub_blocks[0],
                                            &x_name_list, &inner_outputs);

  VLOG(3) << "inner_inputs length:" << x_name_list.size();
  VLOG(3) << "inner_outputs length:" << inner_outputs.size();
}
}  // namespace framework
}  // namespace paddle
