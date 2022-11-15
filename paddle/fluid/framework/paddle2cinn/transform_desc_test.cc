// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/paddle2cinn/transform_desc.h"

#include <unordered_map>

#include "gtest/gtest.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using PbVarType = framework::proto::VarType;
namespace cpp = ::cinn::frontend::paddle::cpp;

// check VarDesc
cpp::VarDesc CreateCppVarDesc() {
  cpp::VarDesc var("test");
  var.SetType(cpp::VarDescAPI::Type::LOD_TENSOR);
  var.SetPersistable(true);
  var.SetDataType(cpp::VarDescAPI::Type::FP32);
  var.SetShape({100, 200, 300});
  return var;
}

framework::VarDesc CreatePbVarDesc() {
  framework::VarDesc var("test");
  var.SetType(PbVarType::LOD_TENSOR);
  var.SetPersistable(true);
  var.SetDataType(PbVarType::FP32);
  var.SetShape({100, 200, 300});
  return var;
}

TEST(TransformVarDesc, cpp2pb) {
  auto cpp_var = CreateCppVarDesc();
  framework::VarDesc pb_var("init");
  TransformVarDescFromCinn(cpp_var, &pb_var);

  auto correct_var = CreatePbVarDesc();
  ASSERT_EQ(pb_var.Name(), correct_var.Name());
  ASSERT_EQ(pb_var.GetType(), correct_var.GetType());
  ASSERT_EQ(pb_var.Persistable(), correct_var.Persistable());
  ASSERT_EQ(pb_var.GetDataType(), correct_var.GetDataType());
  ASSERT_EQ(pb_var.GetShape(), correct_var.GetShape());
}

TEST(TransformVarDesc, pb2cpp) {
  auto pb_var = CreatePbVarDesc();
  cpp::VarDesc cpp_var;
  TransformVarDescToCinn(&pb_var, &cpp_var);

  auto correct_var = CreateCppVarDesc();
  ASSERT_EQ(cpp_var.Name(), correct_var.Name());
  ASSERT_EQ(cpp_var.GetType(), correct_var.GetType());
  ASSERT_EQ(cpp_var.Persistable(), correct_var.Persistable());
  ASSERT_EQ(cpp_var.GetDataType(), correct_var.GetDataType());
  ASSERT_EQ(cpp_var.GetShape(), correct_var.GetShape());
}

// check OpDesc
cpp::OpDesc CreateCppOpDesc() {
  cpp::OpDesc op;
  op.SetType("test");
  op.SetInput("X", {"x1"});
  op.SetInput("Y", {"y1", "y2"});
  op.SetOutput("Out", {"out1"});
  op.SetAttr<float>("attr_f", 0.1f);
  op.SetAttr<std::string>("attr_str", "test_attr");
  return op;
}

framework::OpDesc CreatePbOpDesc() {
  framework::OpDesc op;
  op.SetType("test");
  op.SetInput("X", {"x1"});
  op.SetInput("Y", {"y1", "y2"});
  op.SetOutput("Out", {"out1"});
  op.SetAttr("attr_f", 0.1f);
  op.SetAttr("attr_str", std::string("test_attr"));
  return op;
}

TEST(TransformOpDesc, cpp2pb) {
  auto cpp_op = CreateCppOpDesc();
  framework::OpDesc pb_op;
  TransformOpDescFromCinn(cpp_op, &pb_op);

  auto correct_op = CreatePbOpDesc();
  ASSERT_EQ(pb_op.Type(), correct_op.Type());
  ASSERT_EQ(pb_op.Inputs(), correct_op.Inputs());
  ASSERT_EQ(pb_op.Outputs(), correct_op.Outputs());
  ASSERT_EQ(pb_op.AttrNames(), correct_op.AttrNames());

  for (const auto &attr_name : pb_op.AttrNames()) {
    ASSERT_EQ(pb_op.GetAttrType(attr_name), correct_op.GetAttrType(attr_name));
  }
  ASSERT_EQ(pb_op.GetAttrIfExists<float>("attr_f"),
            correct_op.GetAttrIfExists<float>("attr_f"));
  ASSERT_EQ(pb_op.GetAttrIfExists<std::string>("attr_str"),
            correct_op.GetAttrIfExists<std::string>("attr_str"));
}

TEST(TransformOpDesc, pb2cpp) {
  auto pb_op = CreatePbOpDesc();
  cpp::OpDesc cpp_op;
  TransformOpDescToCinn(&pb_op, &cpp_op);

  auto correct_op = CreateCppOpDesc();
  ASSERT_EQ(cpp_op.Type(), correct_op.Type());
  ASSERT_EQ(cpp_op.inputs(), correct_op.inputs());
  ASSERT_EQ(cpp_op.outputs(), correct_op.outputs());
  ASSERT_EQ(cpp_op.AttrNames(), correct_op.AttrNames());
  ASSERT_EQ(cpp_op.attr_types(), correct_op.attr_types());

  ASSERT_EQ(cpp_op.GetAttr<float>("attr_f"),
            correct_op.GetAttr<float>("attr_f"));
  ASSERT_EQ(cpp_op.GetAttr<std::string>("attr_str"),
            correct_op.GetAttr<std::string>("attr_str"));
}

// check BlockDesc
// framework::BlockDesc is DISABLE_COPY_AND_ASSIGN, so can not return
void CreateCppBlockDesc(cpp::BlockDesc *block) {
  block->SetIdx(42);
  block->SetParentIdx(4);
  block->SetForwardBlockIdx(32);

  auto *op = block->AddOp<cpp::OpDesc>();
  *op = CreateCppOpDesc();

  auto *var = block->AddVar<cpp::VarDesc>();
  *var = CreateCppVarDesc();
}

void CreatePbBlockDesc(framework::BlockDesc *block) {
  block->Proto()->set_idx(42);
  block->Proto()->set_parent_idx(4);
  block->Proto()->set_forward_block_idx(32);

  auto *op = block->AppendOp();
  *op = CreatePbOpDesc();

  auto *var = block->Var("init");
  *var = CreatePbVarDesc();
}

TEST(TransformBlockDesc, cpp2pb) {
  cpp::BlockDesc cpp_block;
  CreateCppBlockDesc(&cpp_block);

  framework::ProgramDesc pb_prog;
  auto *pb_block = pb_prog.MutableBlock(0);
  TransformBlockDescFromCinn(cpp_block, pb_block);

  framework::ProgramDesc correct_prog;
  auto *correct_block = correct_prog.MutableBlock(0);
  CreatePbBlockDesc(correct_block);
  ASSERT_EQ(pb_block->ID(), correct_block->ID());
  ASSERT_EQ(pb_block->Parent(), correct_block->Parent());
  ASSERT_EQ(pb_block->ForwardBlockID(), correct_block->ForwardBlockID());
  ASSERT_EQ(pb_block->OpSize(), correct_block->OpSize());
  ASSERT_EQ(pb_block->AllVars().size(), correct_block->AllVars().size());
}

TEST(TransformBlockDesc, pb2cpp) {
  framework::ProgramDesc pb_prog;
  auto *pb_block = pb_prog.MutableBlock(0);
  CreatePbBlockDesc(pb_block);

  cpp::BlockDesc cpp_block;
  TransformBlockDescToCinn(pb_block, &cpp_block);

  cpp::BlockDesc correct_block;
  CreateCppBlockDesc(&correct_block);
  ASSERT_EQ(cpp_block.Idx(), correct_block.Idx());
  ASSERT_EQ(cpp_block.ParentIdx(), correct_block.ParentIdx());
  ASSERT_EQ(cpp_block.ForwardBlockIdx(), correct_block.ForwardBlockIdx());
  ASSERT_EQ(cpp_block.OpsSize(), correct_block.OpsSize());
  ASSERT_EQ(cpp_block.VarsSize(), correct_block.VarsSize());
}

// check ProgramDesc
cpp::ProgramDesc CreateCppProgramDesc() {
  cpp::ProgramDesc prog;
  prog.SetVersion(22);

  auto *block = prog.AddBlock<cpp::BlockDesc>();
  CreateCppBlockDesc(block);

  return prog;
}

framework::ProgramDesc CreatePbProgramDesc() {
  framework::ProgramDesc prog;
  prog.SetVersion(22);

  auto *block = prog.MutableBlock(0);
  CreatePbBlockDesc(block);
  return prog;
}

TEST(TransformProgramDesc, cpp2pb) {
  auto cpp_prog = CreateCppProgramDesc();
  framework::ProgramDesc pb_prog;
  TransformProgramDescFromCinn(cpp_prog, &pb_prog);

  auto correct_prog = CreatePbProgramDesc();
  ASSERT_EQ(pb_prog.Version(), correct_prog.Version());
  ASSERT_EQ(pb_prog.Size(), correct_prog.Size());
}

TEST(TransformProgramDesc, pb2cpp) {
  auto pb_prog = CreatePbProgramDesc();
  cpp::ProgramDesc cpp_prog;
  TransformProgramDescToCinn(&pb_prog, &cpp_prog);

  auto correct_prog = CreateCppProgramDesc();
  ASSERT_EQ(cpp_prog.Version(), correct_prog.Version());
  ASSERT_EQ(cpp_prog.BlocksSize(), correct_prog.BlocksSize());
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
