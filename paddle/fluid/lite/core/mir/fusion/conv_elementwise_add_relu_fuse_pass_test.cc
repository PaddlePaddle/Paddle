// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/mir/fusion/conv_elementwise_add_activation_fuse_pass.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/mir/passes.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/program.h"

DEFINE_string(model_dir, "", "");
DEFINE_string(optimized_model, "", "");

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

std::unique_ptr<SSAGraph> BuildGraph(framework::ProgramDesc* program_desc,
                                     const std::shared_ptr<Scope>& scope,
                                     const std::vector<Place>& valid_places) {
  auto* main_block = program_desc->MutableBlock(0);

  auto* conv2d_1 = main_block->AppendOp();
  auto* conv2d_2 = main_block->AppendOp();
  auto* add_1 = main_block->AppendOp();
  auto* relu_1 = main_block->AppendOp();
  auto* add_2 = main_block->AppendOp();
  auto* relu_2 = main_block->AppendOp();

  main_block->Var("input_1");
  main_block->Var("input_2");
  main_block->Var("filter_1");
  main_block->Var("filter_2");
  main_block->Var("conv2d_1_out");
  main_block->Var("conv2d_2_out");
  main_block->Var("bias_1");
  main_block->Var("add_1_out");
  main_block->Var("add_2_out");
  main_block->Var("relu_1_out");
  main_block->Var("out");

  scope->Var("input_1")->GetMutable<lite::Tensor>();
  scope->Var("input_2")->GetMutable<lite::Tensor>();
  scope->Var("filter_1")->GetMutable<lite::Tensor>();
  scope->Var("filter_2")->GetMutable<lite::Tensor>();
  scope->Var("conv2d_1_out")->GetMutable<lite::Tensor>();
  scope->Var("conv2d_2_out")->GetMutable<lite::Tensor>();
  scope->Var("bias_1")->GetMutable<lite::Tensor>();
  scope->Var("add_1_out")->GetMutable<lite::Tensor>();
  scope->Var("add_2_out")->GetMutable<lite::Tensor>();
  scope->Var("relu_1_out")->GetMutable<lite::Tensor>();
  scope->Var("out")->GetMutable<lite::Tensor>();

  conv2d_1->SetType("conv2d");
  conv2d_1->SetInput("Input", {"input_1"});
  conv2d_1->SetInput("Filter", {"filter_1"});
  conv2d_1->SetOutput("Output", {"conv2d_1_out"});
  conv2d_1->SetAttr("strides", std::vector<int>({1, 1}));
  conv2d_1->SetAttr("paddings", std::vector<int>({0, 0}));
  conv2d_1->SetAttr("groups", 1);
  conv2d_1->SetAttr("dilations", std::vector<int>({1, 1}));
  conv2d_1->SetAttr("fuse_relu", false);

  add_1->SetType("elementwise_add");
  add_1->SetInput("X", {"conv2d_1_out"});
  add_1->SetInput("Y", {"bias_1"});
  add_1->SetOutput("Out", {"add_1_out"});
  add_1->SetAttr("axis", 1);

  relu_1->SetType("relu");
  relu_1->SetInput("X", {"add_1_out"});
  relu_1->SetOutput("Out", {"relu_1_out"});

  conv2d_2->SetType("conv2d");
  conv2d_2->SetInput("Input", {"input_2"});
  conv2d_2->SetInput("Filter", {"filter_2"});
  conv2d_2->SetOutput("Output", {"conv2d_2_out"});
  conv2d_2->SetAttr("strides", std::vector<int>({1, 1}));
  conv2d_2->SetAttr("paddings", std::vector<int>({0, 0}));
  conv2d_2->SetAttr("groups", 1);
  conv2d_2->SetAttr("dilations", std::vector<int>({1, 1}));
  conv2d_2->SetAttr("fuse_relu", false);

  add_2->SetType("elementwise_add");
  add_2->SetInput("X", {"conv2d_2_out"});
  add_2->SetInput("Y", {"relu_1_out"});
  add_2->SetOutput("Out", {"add_2_out"});
  add_2->SetAttr("axis", 1);

  relu_2->SetType("relu");
  relu_2->SetInput("X", {"add_2_out"});
  relu_2->SetOutput("Out", {"out"});

  program_desc->Flush();

  lite::Program program(*program_desc->Proto(), scope, valid_places);
  auto graph = std::unique_ptr<SSAGraph>(new SSAGraph());
  graph->Build(program, valid_places);

  return graph;
}

TEST(conv_elementwise_add_relu_fuse_pass, graph_test) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);

  Visualize(graph.get());
  ASSERT_EQ(graph->nodes().size(), 11UL /*vars*/ + 6UL /*ops*/);
  Visualize(graph.get());
}

TEST(conv_elementwise_add_relu_fuse_pass, fuse_test_op) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);
  Visualize(graph.get());
  const int num_nodes = graph->nodes().size();
  auto* fuser = new ConvElementwiseAddReLUFusePass;
  fuser->Apply(graph);
  Visualize(graph.get());
  ASSERT_EQ(graph->nodes().size(), num_nodes - 5UL * 2 /*nodes removed */ +
                                       1UL * 2 /* fused fc node*/);
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(elementwise_add);
USE_LITE_OP(conv2d);
USE_LITE_OP(depthwise_conv2d);
USE_LITE_OP(relu);
