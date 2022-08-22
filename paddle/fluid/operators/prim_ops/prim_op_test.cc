// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"

USE_OP_ITSELF(reshape_p);
USE_OP_ITSELF(broadcast_p);
USE_OP_ITSELF(reduce_p);
USE_OP_ITSELF(transpose_p);
USE_OP_ITSELF(split_p);
USE_OP_ITSELF(concat_p);
USE_OP_ITSELF(slice_select_p);
USE_OP_ITSELF(slice_assign_p);
USE_OP_ITSELF(gather_p);
USE_OP_ITSELF(scatter_add_p);
USE_OP_ITSELF(add_p);
USE_OP_ITSELF(sub_p);
USE_OP_ITSELF(mul_p);
USE_OP_ITSELF(div_p);
USE_OP_ITSELF(sqrt_p);
USE_OP_ITSELF(tanh_p);
USE_OP_ITSELF(matmul_p);
USE_OP_ITSELF(fill_constant_p);
USE_OP_ITSELF(log_p);
USE_OP_ITSELF(select_p);
USE_OP_ITSELF(eq_p);
USE_OP_ITSELF(pow_p);
USE_OP_ITSELF(max_p);

namespace paddle {
namespace framework {

static void NewVar(BlockDesc *block,
                   const std::string &name,
                   const std::vector<int64_t> &shape) {
  auto *var_desc = block->Var(name);
  if (shape.size() > 0) {
    var_desc->SetShape(shape);
    var_desc->SetType(proto::VarType::LOD_TENSOR);
    var_desc->SetDataType(proto::VarType_Type_FP32);
  }
}

static void AppendOp(BlockDesc *block,
                     const std::string &type,
                     VariableNameMap inputs,
                     VariableNameMap outputs,
                     AttributeMap attrs) {
  auto &op_info = OpInfoMap::Instance().Get(type);
  if (op_info.Checker()) {
    op_info.Checker()->Check(&attrs);
  }

  auto *op = block->AppendOp();
  op->SetType(type);
  for (auto &pair : inputs) {
    op->SetInput(pair.first, pair.second);
  }

  for (auto &pair : outputs) {
    op->SetOutput(pair.first, pair.second);
    for (auto &var_name : pair.second) {
      if (!block->FindVarRecursive(var_name)) {
        NewVar(block, var_name, {});
      }
    }
  }

  op->SetAttrMap(attrs);
  op->InferVarType(block);
  op->InferShape(*block);
}

TEST(PrimOp, reshape_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block,
           "reshape_p",
           {{"X", {x0}}},
           {{"Y", {x1}}},
           {{"shape", std::vector<int64_t>{12, 5}}});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 2UL);
  ASSERT_EQ(shapes[0], 12L);
  ASSERT_EQ(shapes[1], 5L);
}

TEST(PrimOp, broadcast_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 1};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block,
           "broadcast_p",
           {{"X", {x0}}},
           {{"Y", {x1}}},
           {{"shape", std::vector<int64_t>{3, 4, 5}}});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, reduce_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape);
  AppendOp(block,
           "reduce_p",
           {{"X", {x0}}},
           {{"Y", {x1}}},
           {{"axis", std::vector<int64_t>{0, 2}}, {"keepdim", false}});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 1UL);
  ASSERT_EQ(shapes[0], 4L);
  AppendOp(block,
           "reduce_p",
           {{"X", {x0}}},
           {{"Y", {x2}}},
           {{"axis", std::vector<int64_t>{0, 2}}, {"keepdim", true}});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 1L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 1L);
}

TEST(PrimOp, transpose_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block,
           "transpose_p",
           {{"X", {x0}}},
           {{"Y", {x1}}},
           {{"axis", std::vector<int64_t>{2, 1, 0}}});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 5L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 3L);
}

TEST(PrimOp, split_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{6, 8, 10};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";
  std::string x3 = "x3";

  NewVar(block, x0, shape);
  AppendOp(block,
           "split_p",
           {{"X", {x0}}},
           {{"YS", {x1, x2, x3}}},
           {{"axis", int64_t{1}},
            {"num_or_sections", std::vector<int64_t>{2, 4, 2}}});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 2L);
  ASSERT_EQ(shapes[2], 10L);
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 10L);
  ASSERT_EQ(block->Var("x3")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x3")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x3")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 2L);
  ASSERT_EQ(shapes[2], 10L);
  std::string x4 = "x4";
  std::string x5 = "x5";
  AppendOp(
      block,
      "split_p",
      {{"X", {x0}}},
      {{"YS", {x4, x5}}},
      {{"axis", int64_t{2}}, {"num_or_sections", std::vector<int64_t>{2}}});
  ASSERT_EQ(block->Var("x4")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x4")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x4")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 8L);
  ASSERT_EQ(shapes[2], 5L);
  ASSERT_EQ(block->Var("x5")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x5")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x5")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 8L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, concat_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape_0{3, 1, 5};
  std::vector<int64_t> shape_1{3, 4, 5};
  std::vector<int64_t> shape_2{3, 6, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";
  std::string x3 = "x3";

  NewVar(block, x0, shape_0);
  NewVar(block, x1, shape_1);
  NewVar(block, x2, shape_2);
  AppendOp(block,
           "concat_p",
           {{"XS", {x0, x1, x2}}},
           {{"Y", {x3}}},
           {{"axis", int64_t{1}}});
  ASSERT_EQ(block->Var("x3")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x3")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x3")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 11L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, slice_select_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{6, 8, 10};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block,
           "slice_select_p",
           {{"X", {x0}}},
           {{"Y", {x1}}},
           {{"axis", std::vector<int64_t>{0, 1, 2}},
            {"starts", std::vector<int64_t>{0, 0, 0}},
            {"ends", std::vector<int64_t>{5, 7, 9}},
            {"strides", std::vector<int64_t>{2, 2, 2}}});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, slice_assign_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape_0{6, 8, 10};
  std::vector<int64_t> shape_1{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape_0);
  NewVar(block, x1, shape_1);
  AppendOp(block,
           "slice_assign_p",
           {{"X", {x0}}, {"Y", {x1}}},
           {{"Z", {x2}}},
           {{"axis", std::vector<int64_t>{0, 1, 2}},
            {"starts", std::vector<int64_t>{0, 0, 0}},
            {"ends", std::vector<int64_t>{5, 7, 9}},
            {"strides", std::vector<int64_t>{2, 2, 2}}});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 8L);
  ASSERT_EQ(shapes[2], 10L);
}

TEST(PrimOp, gather_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{6, 8, 10};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block,
           "gather_p",
           {{"X", {x0}}},
           {{"Y", {x1}}},
           {{"axis", int64_t{1}}, {"index", std::vector<int64_t>{0, 2, 5}}});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 3L);
  ASSERT_EQ(shapes[2], 10L);
  std::string index_t = "index_t";
  std::string x2 = "x2";

  auto *var_desc = block->Var(index_t);
  var_desc->SetShape(std::vector<int64_t>{3});
  var_desc->SetType(proto::VarType::LOD_TENSOR);
  var_desc->SetDataType(proto::VarType_Type_INT32);
  AppendOp(block,
           "gather_p",
           {{"X", {x0}}, {"IndexTensor", {index_t}}},
           {{"Y", {x2}}},
           {{"axis", int64_t{1}}});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 3L);
  ASSERT_EQ(shapes[2], 10L);
}

TEST(PrimOp, scatter_add_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape_0{6, 8, 10};
  std::vector<int64_t> shape_1{6, 3, 10};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape_0);
  NewVar(block, x1, shape_1);
  AppendOp(block,
           "scatter_add_p",
           {{"X", {x0}}, {"Y", {x1}}},
           {{"Z", {x2}}},
           {{"axis", int64_t{1}}, {"index", std::vector<int64_t>{0, 2, 5}}});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 8L);
  ASSERT_EQ(shapes[2], 10L);
  std::string index_t = "index_t";
  std::string x3 = "x3";

  auto *var_desc = block->Var(index_t);
  var_desc->SetShape(std::vector<int64_t>{3});
  var_desc->SetType(proto::VarType::LOD_TENSOR);
  var_desc->SetDataType(proto::VarType_Type_INT32);
  AppendOp(block,
           "scatter_add_p",
           {{"X", {x0}}, {"Y", {x1}}, {"IndexTensor", {index_t}}},
           {{"Z", {x3}}},
           {{"axis", int64_t{1}}});
  ASSERT_EQ(block->Var("x3")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x3")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x3")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 6L);
  ASSERT_EQ(shapes[1], 8L);
  ASSERT_EQ(shapes[2], 10L);
}

TEST(PrimOp, add_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape);
  NewVar(block, x1, shape);
  AppendOp(block, "add_p", {{"X", {x0}}, {"Y", {x1}}}, {{"Z", {x2}}}, {});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, sub_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape);
  NewVar(block, x1, shape);
  AppendOp(block, "sub_p", {{"X", {x0}}, {"Y", {x1}}}, {{"Z", {x2}}}, {});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, mul_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape);
  NewVar(block, x1, shape);
  AppendOp(block, "mul_p", {{"X", {x0}}, {"Y", {x1}}}, {{"Z", {x2}}}, {});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, div_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape);
  NewVar(block, x1, shape);
  AppendOp(block, "div_p", {{"X", {x0}}, {"Y", {x1}}}, {{"Z", {x2}}}, {});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, sqrt_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block, "sqrt_p", {{"X", {x0}}}, {{"Y", {x1}}}, {});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, tanh_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block, "tanh_p", {{"X", {x0}}}, {{"Y", {x1}}}, {});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, matmul_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape_0{3, 4, 5};
  std::vector<int64_t> shape_1{3, 5, 8};

  std::string x0 = "x0";
  std::string x1 = "x1";
  std::string x2 = "x2";

  NewVar(block, x0, shape_0);
  NewVar(block, x1, shape_1);
  AppendOp(block, "matmul_p", {{"X", {x0}}, {"Y", {x1}}}, {{"Z", {x2}}}, {});
  ASSERT_EQ(block->Var("x2")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x2")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x2")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 8L);
  std::vector<int64_t> shape_2{4, 5};
  std::vector<int64_t> shape_3{5, 8};

  std::string x3 = "x3";
  std::string x4 = "x4";
  std::string x5 = "x5";

  NewVar(block, x3, shape_2);
  NewVar(block, x4, shape_3);
  AppendOp(block, "matmul_p", {{"X", {x3}}, {"Y", {x4}}}, {{"Z", {x5}}}, {});
  ASSERT_EQ(block->Var("x5")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x5")->GetDataType(), proto::VarType_Type_FP32);
  shapes = block->Var("x5")->GetShape();
  ASSERT_EQ(shapes.size(), 2UL);
  ASSERT_EQ(shapes[0], 4L);
  ASSERT_EQ(shapes[1], 8L);
}

TEST(PrimOp, fill_constant_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::string x0 = "x0";

  AppendOp(block,
           "fill_constant_p",
           {{}},
           {{"Y", {x0}}},
           {{"value", 0.0f},
            {"dtype", proto::VarType_Type_FP32},
            {"shape", std::vector<int64_t>{3, 4, 5}}});
  ASSERT_EQ(block->Var("x0")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x0")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x0")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, log_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x0 = "x0";
  std::string x1 = "x1";

  NewVar(block, x0, shape);
  AppendOp(block, "log_p", {{"X", {x0}}}, {{"Y", {x1}}}, {});
  ASSERT_EQ(block->Var("x1")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("x1")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("x1")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, select_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{2, 3};

  std::string cond = "cond";
  std::string x = "x";
  std::string y = "y";
  std::string z = "z";

  NewVar(block, cond, shape);
  NewVar(block, x, shape);
  NewVar(block, y, shape);

  AppendOp(block,
           "select_p",
           {{"Condition", {cond}}, {"X", {x}}, {"Y", {y}}},
           {{"Z", {z}}},
           {});
  ASSERT_EQ(block->Var("z")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("z")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("z")->GetShape();
  ASSERT_EQ(shapes.size(), 2UL);
  ASSERT_EQ(shapes[0], 2L);
  ASSERT_EQ(shapes[1], 3L);
}

TEST(PrimOp, eq_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x = "x";
  std::string y = "y";
  std::string z = "z";

  NewVar(block, x, shape);
  NewVar(block, y, shape);
  AppendOp(block, "eq_p", {{"X", {x}}, {"Y", {y}}}, {{"Z", {z}}}, {});
  ASSERT_EQ(block->Var("z")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("z")->GetDataType(), proto::VarType::BOOL);
  auto shapes = block->Var("z")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, pow_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{3, 4, 5};

  std::string x = "x";
  std::string y = "y";
  std::string z = "z";

  NewVar(block, x, shape);
  NewVar(block, y, shape);
  AppendOp(block, "pow_p", {{"X", {x}}, {"Y", {y}}}, {{"Z", {z}}}, {});
  ASSERT_EQ(block->Var("z")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("z")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("z")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 3L);
  ASSERT_EQ(shapes[1], 4L);
  ASSERT_EQ(shapes[2], 5L);
}

TEST(PrimOp, max_p) {
  ProgramDesc program;
  auto *block = program.MutableBlock(0);
  std::vector<int64_t> shape{2, 3, 4};

  std::string x = "x";
  std::string y = "y";
  std::string z = "z";

  NewVar(block, x, shape);
  NewVar(block, y, shape);

  AppendOp(block, "max_p", {{"X", {x}}, {"Y", {y}}}, {{"Z", {z}}}, {});
  ASSERT_EQ(block->Var("z")->GetType(), proto::VarType::LOD_TENSOR);
  ASSERT_EQ(block->Var("z")->GetDataType(), proto::VarType_Type_FP32);
  auto shapes = block->Var("z")->GetShape();
  ASSERT_EQ(shapes.size(), 3UL);
  ASSERT_EQ(shapes[0], 2L);
  ASSERT_EQ(shapes[1], 3L);
  ASSERT_EQ(shapes[2], 4L);
}

}  // namespace framework
}  // namespace paddle
