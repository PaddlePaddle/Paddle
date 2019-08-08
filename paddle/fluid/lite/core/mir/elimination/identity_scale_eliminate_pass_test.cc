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

#include <gtest/gtest.h>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"
#include "paddle/fluid/lite/core/mir/ssa_graph.h"

namespace paddle {
namespace lite {
namespace mir {

std::unique_ptr<SSAGraph> BuildGraph(framework::ProgramDesc* program_desc,
                                     const std::shared_ptr<Scope>& scope,
                                     const std::vector<Place>& valid_places) {
  // Op list:
  // (x)->feed -> (feed) -> scale -> (scale_out) -> fetch->(fetch)
  // After pass
  // (x)->feed->(scale_out)->fetch->(fetch)
  auto* main_block = program_desc->MutableBlock(0);
  auto* feed_op = main_block->AppendOp();
  auto* scale_op = main_block->AppendOp();
  auto* fetch_op = main_block->AppendOp();
  main_block->Var("x");
  main_block->Var("feed");
  main_block->Var("scale_out");
  main_block->Var("fetch_out");

  scope->Var("x")->GetMutable<lite::Tensor>();
  scope->Var("feed")->GetMutable<lite::Tensor>();
  scope->Var("scale_out")->GetMutable<lite::Tensor>();
  scope->Var("fetch_out")->GetMutable<lite::Tensor>();

  feed_op->SetType("feed");
  feed_op->SetInput("X", {"x"});
  feed_op->SetAttr("col", 1);
  feed_op->SetOutput("Out", {"feed"});

  scale_op->SetType("scale");
  scale_op->SetInput("X", {"feed"});
  scale_op->SetOutput("Out", {"scale_out"});
  scale_op->SetAttr("scale", 1.f);
  scale_op->SetAttr("bias", 0.f);
  scale_op->SetAttr("bias_after_scale", true);

  fetch_op->SetType("fetch");
  fetch_op->SetInput("X", {"scale_out"});
  fetch_op->SetOutput("Out", {"fetch"});
  fetch_op->SetAttr("col", 1);

  program_desc->Flush();

  lite::Program program(*program_desc->Proto(), scope, valid_places);
  auto graph = std::unique_ptr<SSAGraph>(new SSAGraph());
  graph->Build(program, valid_places);

  LOG(INFO) << Visualize(graph.get());

  return graph;
}

TEST(identity_test, test) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);
  const int num_nodes = graph->nodes().size();
  auto pass = PassManager::Global().LookUp("identity_scale_eliminate_pass");
  ASSERT_TRUE(pass);
  pass->Apply(graph);
  ASSERT_EQ(graph->nodes().size(), num_nodes - 2UL);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(feed)
USE_LITE_OP(fetch)
USE_LITE_OP(scale)
USE_MIR_PASS(identity_scale_eliminate_pass)
