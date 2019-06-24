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

#include "paddle/fluid/lite/core/mir/fusion/elementwise_add_activation_fuse_pass.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/program.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

std::unique_ptr<SSAGraph> BuildGraph(framework::ProgramDesc* program_desc,
                                     const std::shared_ptr<Scope>& scope,
                                     const std::vector<Place>& valid_places) {
  auto* main_block = program_desc->MutableBlock(0);

  auto* add_1 = main_block->AppendOp();
  auto* add_2 = main_block->AppendOp();
  auto* relu_1 = main_block->AppendOp();
  auto* relu_2 = main_block->AppendOp();

  main_block->Var("x_1");
  main_block->Var("y_1");
  main_block->Var("add_out_1");
  main_block->Var("relu_out_1");
  main_block->Var("y_2");
  main_block->Var("add_out_2");
  main_block->Var("out");

  scope->Var("x_1")->GetMutable<lite::Tensor>();
  scope->Var("y_1")->GetMutable<lite::Tensor>();
  scope->Var("add_out_1")->GetMutable<lite::Tensor>();
  scope->Var("relu_out_1")->GetMutable<lite::Tensor>();
  scope->Var("y_2")->GetMutable<lite::Tensor>();
  scope->Var("add_out_2")->GetMutable<lite::Tensor>();
  scope->Var("out")->GetMutable<lite::Tensor>();

  add_1->SetType("elementwise_add");
  add_1->SetInput("X", {"x_1"});
  add_1->SetInput("Y", {"y_1"});
  add_1->SetOutput("Out", {"add_out_1"});
  add_1->SetAttr("axis", 1);

  relu_1->SetType("relu");
  relu_1->SetInput("X", {"add_out_1"});
  relu_1->SetOutput("Out", {"relu_out_1"});

  add_2->SetType("elementwise_add");
  add_2->SetInput("X", {"relu_out_1"});
  add_2->SetInput("Y", {"y_2"});
  add_2->SetOutput("Out", {"add_out_2"});
  add_2->SetAttr("axis", 1);

  relu_2->SetType("relu");
  relu_2->SetInput("X", {"add_out_2"});
  relu_2->SetOutput("Out", {"out"});

  program_desc->Flush();

  lite::Program program(*program_desc->Proto(), scope, valid_places);
  auto graph = std::unique_ptr<SSAGraph>(new SSAGraph());
  graph->Build(program, valid_places);

  return graph;
}

TEST(elementwise_add_activation_fuse_pass, graph_test) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);
  ASSERT_EQ(graph->nodes().size(),
            7UL /*vars*/ + 4UL /*ops*/ + 1UL /* SSAGraph tmp node*/);
}

TEST(elementwise_add_activation_fuse_pass, fuse_test_op) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);
  Visualize(graph.get());
  const int num_nodes = graph->nodes().size();
  auto* fuser = new ElementwiseAddActivationFusePass;
  fuser->Apply(graph);
  Visualize(graph.get());
  ASSERT_EQ(graph->nodes().size(),
            num_nodes - 3UL * 2 /*nodes removed */ + 1UL * 2 /* fused nodes*/);
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(elementwise_add);
USE_LITE_OP(fusion_elementwise_add_activation);
USE_LITE_OP(relu);
