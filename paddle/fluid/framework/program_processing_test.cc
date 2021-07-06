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
  auto* mul_1_x = global_block->Var("Mul_1_X");
  mul_1_x->SetType(proto::VarType::LOD_TENSOR);
  mul_1_x->SetLoDLevel(0);
  mul_1_x->SetDataType(proto::VarType::FP32);
  mul_1_x->SetShape({1000, 784});

  auto* mul_1_y = global_block->Var("Mul_1_Y");
  mul_1_y->SetType(proto::VarType::LOD_TENSOR);
  mul_1_y->SetLoDLevel(0);
  mul_1_y->SetDataType(proto::VarType::FP32);
  mul_1_y->SetShape({784, 100});

  auto* mul_1_out = global_block->Var("Mul_1_Out");
  mul_1_out->SetType(proto::VarType::LOD_TENSOR);
  auto* mul_op_1 = global_block->AppendOp();

  mul_op_1->SetType("mul");
  mul_op_1->SetInput("X", {mul_1_x->Name()});
  mul_op_1->SetInput("Y", {mul_1_y->Name()});
  mul_op_1->SetOutput("Y", {mul_1_out->Name()});

  // building cond op such as less_than
  auto* less_than_op_1 = global_block->AppendOp();
  less_than_op_1->SetType("less_than");
  auto* less_than_1_x = global_block->Var("Less_than_1_X");
  less_than_1_x->SetType(proto::VarType::LOD_TENSOR);
  less_than_1_x->SetLoDLevel(0);
  less_than_1_x->SetDataType(proto::VarType::FP32);
  less_than_1_x->SetShape({1});

  auto* less_than_1_y = global_block->Var("Less_than_1_Y");
  less_than_1_y->SetType(proto::VarType::LOD_TENSOR);
  less_than_1_y->SetLoDLevel(0);
  less_than_1_y->SetDataType(proto::VarType::FP32);
  less_than_1_y->SetShape({1});

  auto* less_than_1_out = global_block->Var("Less_than_1_Out");
  less_than_1_out->SetType(proto::VarType::BOOL);

  less_than_op_1->SetInput("X", {less_than_1_x->Name()});
  less_than_op_1->SetInput("Y", {less_than_1_y->Name()});
  less_than_op_1->SetOutput("Out", {less_than_1_out->Name()});

  BlockDesc* sub_block = program.AppendBlock(*global_block);
  std::vector<BlockDesc*> sub_blocks;
  sub_blocks.push_back(sub_block);

  BlockDesc* sub_block2 =
      program.AppendBlock(*sub_block);  // for testing nested case.
  sub_blocks.push_back(sub_block2);

  // building while op in sub_block
  auto* while_op = global_block->AppendOp();
  while_op->SetType("while");
  while_op->SetAttr("sub_block", sub_blocks[0]);

  auto* while_x = global_block->Var("While_X");
  while_x->SetType(proto::VarType::LOD_TENSOR);
  while_x->SetLoDLevel(0);
  while_x->SetDataType(proto::VarType::FP32);
  while_x->SetShape({1});

  while_op->SetInput("kX", {while_x->Name()});
  while_op->SetInput("kCondition", {less_than_1_out->Name()});

  auto* while_out = global_block->Var("While_Out");
  while_out->SetType(proto::VarType::LOD_TENSOR);
  while_out->SetLoDLevel(0);
  while_out->SetDataType(proto::VarType::FP32);
  while_out->SetShape({1});

  auto* steps = global_block->Var("StepScopes");

  while_op->SetOutput("kOutputs", {while_out->Name()});
  while_op->SetOutput("kStepScopes", {steps->Name()});

  auto* mul_2_x = global_block->Var("Mul_2_X");
  mul_2_x->SetType(proto::VarType::LOD_TENSOR);
  mul_2_x->SetLoDLevel(0);
  mul_2_x->SetDataType(proto::VarType::FP32);
  mul_2_x->SetShape({1000, 784});

  auto* mul_2_y = global_block->Var("Mul_2_Y");
  mul_2_y->SetType(proto::VarType::LOD_TENSOR);
  mul_2_y->SetLoDLevel(0);
  mul_2_y->SetDataType(proto::VarType::FP32);
  mul_2_y->SetShape({784, 100});

  auto* mul_op_2 = sub_blocks[0]->AppendOp();
  mul_op_2->SetType("mul");
  mul_op_2->SetInput("X", {mul_2_x->Name()});
  mul_op_2->SetInput("Y", {mul_2_y->Name()});

  auto* mul_2_out = global_block->Var("Mul_2_Out");
  mul_2_out->SetType(proto::VarType::LOD_TENSOR);
  mul_op_2->SetOutput("Y", {mul_2_out->Name()});

  auto* less_than_op_2 = sub_blocks[0]->AppendOp();
  less_than_op_2->SetType("less_than");
  auto* less_than_2_x = global_block->Var("Less_than_2_X");
  less_than_2_x->SetType(proto::VarType::LOD_TENSOR);
  less_than_2_x->SetLoDLevel(0);
  less_than_2_x->SetDataType(proto::VarType::FP32);
  less_than_2_x->SetShape({1});

  auto* less_than_2_y = global_block->Var("Less_than_2_Y");
  less_than_2_y->SetType(proto::VarType::LOD_TENSOR);
  less_than_2_y->SetLoDLevel(0);
  less_than_2_y->SetDataType(proto::VarType::FP32);
  less_than_2_y->SetShape({1});

  less_than_op_2->SetInput("X", {less_than_2_x->Name()});
  less_than_op_2->SetInput("Y", {less_than_2_y->Name()});

  auto* less_than_2_out = global_block->Var("Less_than_2_Out");
  less_than_2_out->SetType(proto::VarType::BOOL);
  less_than_op_2->SetOutput("Out", {less_than_2_out->Name()});

  auto* cond_op = sub_blocks[0]->AppendOp();
  cond_op->SetType("conditional_block");
  cond_op->SetAttr("sub_block", sub_blocks[1]);

  auto* cond_x = sub_blocks[0]->Var("Cond_X");
  cond_x->SetType(proto::VarType::LOD_TENSOR);
  cond_x->SetLoDLevel(0);
  cond_x->SetDataType(proto::VarType::FP32);
  cond_x->SetShape({1});

  cond_op->SetInput("kInputs", {cond_x->Name()});
  cond_op->SetInput("kCondition", {less_than_2_out->Name()});

  auto* cond_out = sub_blocks[0]->Var("Cond_Out");
  cond_out->SetType(proto::VarType::LOD_TENSOR);
  cond_out->SetLoDLevel(0);
  cond_out->SetDataType(proto::VarType::FP32);
  cond_out->SetShape({1});

  auto* scope = sub_blocks[0]->Var("Scope");
  scope->SetType(proto::VarType::STEP_SCOPES);

  cond_op->SetOutput("kOutputs", {cond_out->Name()});
  cond_op->SetOutput("kScope", {scope->Name()});

  auto* mul_3_x = global_block->Var("Mul_3_X");
  mul_3_x->SetType(proto::VarType::LOD_TENSOR);
  mul_3_x->SetLoDLevel(0);
  mul_3_x->SetDataType(proto::VarType::FP32);
  mul_3_x->SetShape({1000, 784});

  auto* mul_3_y = global_block->Var("Mul_3_Y");
  mul_3_y->SetType(proto::VarType::LOD_TENSOR);
  mul_3_y->SetLoDLevel(0);
  mul_3_y->SetDataType(proto::VarType::FP32);
  mul_3_y->SetShape({784, 100});

  auto* mul_3_out = global_block->Var("Mul_3_Out");
  mul_3_out->SetType(proto::VarType::LOD_TENSOR);

  auto* mul_op_3 = sub_blocks[1]->AppendOp();
  mul_op_3->SetType("mul");
  mul_op_3->SetInput("X", {mul_3_x->Name()});
  mul_op_3->SetInput("Y", {mul_3_y->Name()});
  mul_op_3->SetOutput("Y", {mul_3_out->Name()});

  ProgramProcessor program_processor;
  std::set<std::string> inner_inputs;
  std::set<std::string> inner_outputs;

  program_processor.GetInputsOutputsInBlock(*sub_blocks[0], &inner_inputs,
                                            &inner_outputs);

  VLOG(3) << "inner_inputs().size():" << inner_inputs.size();
  VLOG(3) << "inner_outputs().size():" << inner_outputs.size();

  ASSERT_EQ(5UL, inner_inputs.size());
  ASSERT_EQ(2UL, inner_outputs.size());

  // varible "Less_than_2_Out" is the input of cond_op, it also is the output of
  // less_than_op.
  std::set<std::string> inner_inputs_{"Less_than_2_Out", "Less_than_2_X",
                                      "Less_than_2_Y", "Mul_2_X", "Mul_2_Y"};
  std::set<std::string> inner_outputs_{"Less_than_2_Out", "Mul_2_Out"};

  ASSERT_EQ(inner_inputs, inner_inputs_);
  ASSERT_EQ(inner_outputs, inner_outputs_);

  // Test AddDepToBlockOp
  VLOG(3) << "Before AddDependency, while op's input kX size:"
          << while_op->Input("kX").size();
  VLOG(3) << "Before AddDependency, while op's output kOutPuts size:"
          << while_op->Output("kOutputs").size();

  program_processor.AddDepToBlockOp(*global_block);

  VLOG(3) << "After AddDependency, while op's input kX size:"
          << while_op->Input("kX").size();
  VLOG(3) << "After AddDependency, while op's output kOutPuts size:"
          << while_op->Output("kOutputs").size();

  ASSERT_EQ(8UL, while_op->Input("kX").size());
  ASSERT_EQ(4UL, while_op->Output("kOutputs").size());

  std::vector<std::string> var_input_vec = {
      "While_X", "Less_than_2_Out", "Less_than_2_X", "Less_than_2_Y",
      "Mul_2_X", "Mul_2_Y",         "Mul_3_X",       "Mul_3_Y"};

  std::vector<std::string> var_output_vec = {"While_Out", "Less_than_2_Out",
                                             "Mul_2_Out", "Mul_3_Out"};

  ASSERT_EQ(var_input_vec, while_op->Input("kX"));
  ASSERT_EQ(var_output_vec, while_op->Output("kOutputs"));
}
}  // namespace framework
}  // namespace paddle
