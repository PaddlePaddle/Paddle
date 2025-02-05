// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

VarDesc* Data(paddle::framework::BlockDesc* block,
              std::string name,
              std::vector<int64_t> shape = {},
              bool is_persistable = false,
              proto::VarType::Type data_type = proto::VarType::FP32) {
  auto* var = block->Var(name);
  var->SetType(proto::VarType::DENSE_TENSOR);
  var->SetDataType(data_type);
  var->SetShape(shape);
  var->SetPersistable(is_persistable);
  return var;
}

VarDesc* fill_constant(BlockDesc* block, std::vector<VarDesc*> shapes) {
  VarDesc* out = Data(block, shapes[0]->Name() + "_out");
  OpDesc* op = block->AppendOp();
  op->SetType("fill_constant");
  std::vector<std::string> shape_names;
  for (auto shape : shapes) {
    shape_names.push_back(shape->Name());
  }
  op->SetInput("ShapeTensorList", {shape_names});
  op->SetOutput("Out", {out->Name()});
  return out;
}

TEST(FillConstantReshapePass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* shape0 = Data(block, "shape0");
  auto* shape1 = Data(block, "shape1");
  auto* shape2 = Data(block, "shape2");
  auto* shape3 = Data(block, "shape3");
  auto* shape4 = Data(block, "shape4");
  auto* shape5 = Data(block, "shape5");
  auto* shape6 = Data(block, "shape6");
  auto* shape7 = Data(block, "shape7");
  auto* shape8 = Data(block, "shape8");
  auto* shape9 = Data(block, "shape9");
  auto* fill0 = fill_constant(block, {shape0, shape1, shape2, shape3, shape4});
  fill0->SetShape({1, 2, 3, 4, 5});
  auto* fill1 = fill_constant(block, {shape5, shape6, shape7, shape8, shape9});
  fill1->SetShape({1, 2, 3, 4, 5});
  OpDesc* fused_multi_transformer = block->AppendOp();
  fused_multi_transformer->SetType("fused_multi_transformer");
  fused_multi_transformer->SetInput("CacheKV", {fill0->Name(), fill1->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto pass = PassRegistry::Instance().Get(
      "fused_multi_transformer_cachekv_layout_trans_pass");
  pass->Apply(graph.get());
  auto fills = GetOpNodes(graph, "fill_constant");
  auto fill0_in_names = fills[0]->Op()->Input("ShapeTensorList");
  std::vector<std::string> expect_fill0_out_names{
      "shape5", "shape6", "shape7", "shape8", "shape9"};
  std::vector<std::string> expect_fill1_out_names{
      "shape0", "shape1", "shape2", "shape3", "shape4"};
  PADDLE_ENFORCE_EQ(fill0_in_names,
                    expect_fill0_out_names,
                    common::errors::PreconditionNotMet(
                        "fill_constant name should not be updated."));
  auto fill1_in_names = fills[1]->Op()->Input("ShapeTensorList");
  PADDLE_ENFORCE_EQ(fill1_in_names,
                    expect_fill1_out_names,
                    common::errors::PreconditionNotMet(
                        "fill_constant name should not be updated."));
}

TEST(GatherReshapePass, basic) {
  Layers layers;
  auto* gather0_x = layers.data("gather0_x", {2, 1, 24, 512, 64});
  auto* gather0_index = layers.data("gather0_index", {1});
  auto* gather0_out = layers.gather(gather0_x, gather0_index, 1);
  gather0_out->SetShape({2, 1, 24, 512, 64});
  auto* gather1_x = layers.data("gather1_x", {2, 1, 24, 512, 64});
  auto* gather1_index = layers.data("gather1_index", {1});
  auto* gather1_out = layers.gather(gather1_x, gather1_index, 1);
  gather1_out->SetShape({2, 1, 24, 512, 64});
  auto* block = layers.Block();
  OpDesc* fused_multi_transformer = block->AppendOp();
  fused_multi_transformer->SetType("fused_multi_transformer");
  fused_multi_transformer->SetInput("CacheKV",
                                    {gather0_out->Name(), gather1_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get(
      "fused_multi_transformer_cachekv_layout_trans_pass");
  pass->Apply(graph.get());
  auto gathers = GetOpNodes(graph, "gather");
  for (auto* gather : gathers) {
    PADDLE_ENFORCE_EQ(gather->Op()->GetAttrIfExists<int>("axis"),
                      1,
                      common::errors::PreconditionNotMet(
                          "gather's axis attr should not be updated by pass."));
  }
}

TEST(FillConstantAndGatherReshapePass, basic) {
  Layers layers;
  auto* block = layers.Block();
  auto* shape0 = Data(block, "shape0");
  auto* shape1 = Data(block, "shape1");
  auto* shape2 = Data(block, "shape2");
  auto* shape3 = Data(block, "shape3");
  auto* shape4 = Data(block, "shape4");
  auto* shape5 = Data(block, "shape5");
  auto* shape6 = Data(block, "shape6");
  auto* shape7 = Data(block, "shape7");
  auto* shape8 = Data(block, "shape8");
  auto* shape9 = Data(block, "shape9");
  auto* fill0 = fill_constant(block, {shape0, shape1, shape2, shape3, shape4});
  fill0->SetShape({1, 2, 3, 4, 5});
  auto* fill1 = fill_constant(block, {shape5, shape6, shape7, shape8, shape9});
  fill1->SetShape({1, 2, 3, 4, 5});
  OpDesc* fused_multi_transformer = block->AppendOp();
  fused_multi_transformer->SetType("fused_multi_transformer");
  fused_multi_transformer->SetInput("CacheKV", {fill0->Name(), fill1->Name()});

  auto* gather0_x = layers.data("gather0_x", {2, 1, 24, 512, 64});
  auto* gather0_index = layers.data("gather0_index", {1});
  auto* gather0_out = layers.gather(gather0_x, gather0_index, 1);
  gather0_out->SetShape({2, 1, 24, 512, 64});
  auto* gather1_x = layers.data("gather1_x", {2, 1, 24, 512, 64});
  auto* gather1_index = layers.data("gather1_index", {1});
  auto* gather1_out = layers.gather(gather1_x, gather1_index, 1);
  gather1_out->SetShape({2, 1, 24, 512, 64});
  OpDesc* fused_multi_transformer1 = block->AppendOp();
  fused_multi_transformer1->SetType("fused_multi_transformer");
  fused_multi_transformer1->SetInput(
      "CacheKV", {gather0_out->Name(), gather1_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get(
      "fused_multi_transformer_cachekv_layout_trans_pass");
  pass->Apply(graph.get());

  auto fills = GetOpNodes(graph, "fill_constant");
  auto fill0_in_names = fills[0]->Op()->Input("ShapeTensorList");
  std::vector<std::string> expect_fill0_out_names{
      "shape0", "shape3", "shape1", "shape2", "shape4"};
  std::vector<std::string> expect_fill1_out_names{
      "shape5", "shape8", "shape6", "shape7", "shape9"};
  PADDLE_ENFORCE_EQ(fill0_in_names,
                    expect_fill0_out_names,
                    common::errors::PreconditionNotMet(
                        "fill_constant name should be updated."));
  auto fill1_in_names = fills[1]->Op()->Input("ShapeTensorList");
  PADDLE_ENFORCE_EQ(fill1_in_names,
                    expect_fill1_out_names,
                    common::errors::PreconditionNotMet(
                        "fill_constant name should be updated."));
  auto gathers = GetOpNodes(graph, "gather");
  for (auto* gather : gathers) {
    PADDLE_ENFORCE_EQ(
        gather->Op()->GetAttrIfExists<int>("axis"),
        2,
        common::errors::PreconditionNotMet(
            "gather's axis attr should be updated to 2 by pass."));
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fused_multi_transformer_cachekv_layout_trans_pass);
