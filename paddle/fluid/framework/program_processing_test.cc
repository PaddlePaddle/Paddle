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

TEST(ProgramDesc, GetInputsOutputsInBlock) {
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

  auto* op1 = global_block->AppendOp();
  op1->SetType("mul");
  op1->SetInput("X", {x->Name()});
  op1->SetInput("Y", {y->Name()});

  auto* out1 = global_block->Var("Out1");
  out1->SetType(proto::VarType::LOD_TENSOR);
  op1->SetOutput("Y", {out1->Name()});

  BlockDesc* new_block = program.AppendBlock(*global_block);
  auto* op2 = new_block->AppendOp();
  op2->SetType("mul");

  auto* op3 = global_block->AppendOp();
  op3->SetType("op_with_subblock");
  op3->SetAttr("sub_block", new_block);
  std::vector<BlockDesc*> sub_blocks;
  sub_blocks.push_back(program.AppendBlock(*global_block));
  op3->SetAttr("sub_blocks", sub_blocks);

  // building cond op such as less_than
  BlockDesc* parent_block = program.MutableBlock(new_block->Parent());
  auto* op4 = parent_block->AppendOp();
  op4->SetType("less_than");
  auto* x1 = parent_block->Var("X1");
  x1->SetType(proto::VarType::LOD_TENSOR);
  x1->SetLoDLevel(0);
  x1->SetDataType(proto::VarType::FP32);
  x1->SetShape({1});

  auto* y1 = parent_block->Var("Y1");
  y1->SetType(proto::VarType::LOD_TENSOR);
  y1->SetLoDLevel(0);
  y1->SetDataType(proto::VarType::FP32);
  y1->SetShape({1});

  op4->SetInput("X", {x1->Name()});
  op4->SetInput("Y", {y1->Name()});

  auto* less_than_out = parent_block->Var("Out1");
  out1->SetType(proto::VarType::BOOL);
  op4->SetOutput("Out", {less_than_out->Name()});

  // building while op in sub_block
  auto* op5 = sub_blocks[0]->AppendOp();
  op5->SetType("while");
  op5->SetAttr("sub_block", sub_blocks[0]);

  auto* x2 = sub_blocks[0]->Var("X2");
  x2->SetType(proto::VarType::LOD_TENSOR);
  x2->SetLoDLevel(0);
  x2->SetDataType(proto::VarType::FP32);
  x2->SetShape({1});

  op5->SetInput("kX", {x2->Name()});
  op5->SetInput("kCondition", {less_than_out->Name()});

  auto* out2 = sub_blocks[0]->Var("Out2");
  out2->SetType(proto::VarType::LOD_TENSOR);
  out2->SetLoDLevel(0);
  out2->SetDataType(proto::VarType::FP32);
  out2->SetShape({1});

  auto* steps = sub_blocks[0]->Var("StepScopes");

  op5->SetOutput("kOutputs", {out2->Name()});
  op5->SetOutput("kStepScopes", {steps->Name()});

  auto* x3 = global_block->Var("X3");
  x3->SetType(proto::VarType::LOD_TENSOR);
  x3->SetLoDLevel(0);
  x3->SetDataType(proto::VarType::FP32);
  x3->SetShape({1000, 784});

  auto* y3 = global_block->Var("Y3");
  y3->SetType(proto::VarType::LOD_TENSOR);
  y3->SetLoDLevel(0);
  y3->SetDataType(proto::VarType::FP32);
  y3->SetShape({784, 100});

  auto* op6 = sub_blocks[0]->AppendOp();
  op6->SetType("mul");
  op6->SetInput("X", {x3->Name()});
  op6->SetInput("Y", {y3->Name()});

  auto* out3 = global_block->Var("Out3");
  out3->SetType(proto::VarType::LOD_TENSOR);
  op6->SetOutput("Y", {out3->Name()});

  ProgramProcessor program_processor;
  VariableNameMap inner_inputs;
  VariableNameMap inner_outputs;

  program_processor.GetInputsOutputsInBlock(*sub_blocks[0], &inner_inputs,
                                            &inner_outputs);

  VLOG(3) << "inner_inputs().size():" << inner_inputs.size();
  VLOG(3) << "inner_outputs().size():" << inner_outputs.size();

  // while op inner_inputs : kCondition = Out1
  ASSERT_EQ(3UL, inner_inputs.size());
  // while op inner_outputs : kOutputs = Out2, kStepScopes = StepScopes

  ASSERT_EQ(3UL, inner_outputs.size());

  VLOG(3) << "Before AddDependency, op's input size:"
          << op5->InputNames().size();
  VLOG(3) << "Before AddDependency, op's output size:"
          << op5->OutputNames().size();
  program_processor.AddDepToBlockOp(*sub_blocks[0]);
  VLOG(3) << "After AddDependency, op's input size:"
          << op5->InputNames().size();
  VLOG(3) << "After AddDependency, op's output size:"
          << op5->OutputNames().size();

  ASSERT_EQ(4UL, op5->InputNames().size());
  ASSERT_EQ(3UL, op5->OutputNames().size());
}
}  // namespace framework
}  // namespace paddle
