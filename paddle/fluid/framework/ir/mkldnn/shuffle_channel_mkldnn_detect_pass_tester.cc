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

#include <gtest/gtest.h>

#include <vector>

#include "paddle/fluid/framework/ir/mkldnn/shuffle_channel_mkldnn_detect_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(phi::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "prog_x", {1, 128, 52, 52});
  return param_scope;
}

void MainTest() {
  Layers layers;
  auto prog_x = layers.data("prog_x", {1, 128, 52, 52});
  auto first_reshape2 = layers.reshape2(prog_x, {-1, 2, 64, 52, 52}, true);
  first_reshape2->SetShape({-1, 2, 64, 52, 52});
  auto transpose2 = layers.transpose2(first_reshape2, {0, 2, 1, 3, 4}, true);
  transpose2->SetShape({-1, 64, 2, 52, 52});
  auto second_reshape2 = layers.reshape2(transpose2, {-1, 128, 52, 52}, true);
  second_reshape2->SetShape({-1, 128, 52, 52});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());

  int added_nodes = 1;    // shuffle_channel
  int removed_nodes = 5;  // 2 * reshape, reshape_out, transpose, transpose_out

  int original_nodes_num = graph->Nodes().size();
  auto pass =
      PassRegistry::Instance().Get("shuffle_channel_mkldnn_detect_pass");
  graph.reset(pass->Apply(graph.release()));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_EQ(current_nodes_num,
            original_nodes_num + added_nodes - removed_nodes);
  EXPECT_EQ(GetNumOpNodes(graph, "reshape2"), 0);
  EXPECT_EQ(GetNumOpNodes(graph, "transpose2"), 0);
  EXPECT_EQ(GetNumOpNodes(graph, "shuffle_channel"), 1);

  for (const auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "shuffle_channel") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(PADDLE_GET_CONST(bool, op->GetAttr("use_mkldnn")));
    }
  }
}

TEST(ShuffleChannelOneDNNDetectPass, ShuffleChannelOneDNNDetectPassTest) {
  MainTest();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(shuffle_channel_mkldnn_detect_pass);
