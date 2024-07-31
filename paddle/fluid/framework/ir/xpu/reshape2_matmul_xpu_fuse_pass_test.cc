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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(Squeeze2MatmulXPUFusePass, basic) {
  Layers layers;

  auto* squeeze2_in = layers.data("squeeze2_in", {64, 1, 74, 1});
  auto* squeeze2_out = layers.squeeze2(squeeze2_in, std::vector<int>{1, 3});
  auto* matmul_y = layers.data("matmul_y", {74, 64}, true);
  auto* matmul_out =
      layers.matmul(squeeze2_out, matmul_y, nullptr, false, false);
  auto* ele_y = layers.data("ele_y", {64}, true);
  layers.elementwise_add(matmul_out, ele_y);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("squeeze2_matmul_xpu_fuse_pass");
  VLOG(3) << DebugString(graph);

  pass->Apply(graph.get());
  VLOG(3) << DebugString(graph);

  auto ops_num = GetNumOpNodes(graph);
  PADDLE_ENFORCE_EQ(
      ops_num,
      3,
      common::errors::PreconditionNotMet(
          "graph should only have 2 op nodes, but received %d.", ops_num));
}

TEST(ReShape2MatmulXPUFusePass, basic) {
  Layers layers;

  auto* reshape2_in = layers.data("reshape2_in", {64, 1, 74, 1});
  auto* reshape2_out = layers.reshape2(reshape2_in, std::vector<int>{-1, 74});
  auto* matmul_y = layers.data("matmul_y", {74, 64}, true);
  auto* matmul_out =
      layers.matmul(reshape2_out, matmul_y, nullptr, false, false);
  auto* ele_y = layers.data("ele_y", {64}, true);
  layers.elementwise_add(matmul_out, ele_y);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("reshape2_matmul_xpu_fuse_pass");
  VLOG(3) << DebugString(graph);

  pass->Apply(graph.get());
  VLOG(3) << DebugString(graph);

  auto ops_num = GetNumOpNodes(graph);
  PADDLE_ENFORCE_EQ(
      ops_num,
      3,
      common::errors::PreconditionNotMet(
          "graph should only have 2 op nodes, but received %d.", ops_num));
}

TEST(MapMatmulV2ToMatmulXPUPass, basic) {
  Layers layers;

  auto* matmul_x = layers.data("matmul_x", {64, 74});
  auto* matmul_y = layers.data("matmul_y", {74, 64}, true);
  layers.matmul_v2(matmul_x, matmul_y, nullptr, false, false);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("map_matmulv2_to_matmul_xpu_pass");
  VLOG(3) << DebugString(graph);

  pass->Apply(graph.get());
  VLOG(3) << DebugString(graph);

  auto matmuls = GetOpNodes(graph, "matmul");
  for (auto* matmul : matmuls) {
    PADDLE_ENFORCE_EQ(
        std::abs(matmul->Op()->GetAttrIfExists<float>("alpha") - 1.f) < 1e-5f,
        true,
        common::errors::PreconditionNotMet(
            "matmul_v2 is mapped to matmul by pass."));
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(reshape2_matmul_xpu_fuse_pass);
