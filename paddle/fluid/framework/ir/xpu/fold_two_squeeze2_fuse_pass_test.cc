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

TEST(FoldTwoSqueeze2FusePass, basic) {
  Layers layers;

  auto* in_x = layers.data("in_x", {64, 1, 74, 1});
  auto* squeeze2_1_out = layers.squeeze2(in_x, std::vector<int>{3});
  layers.squeeze2(squeeze2_1_out, std::vector<int>{1});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("fold_two_squeeze2_fuse_pass");
  pass->Apply(graph.get());
  auto ops_num = GetNumOpNodes(graph);
  PADDLE_ENFORCE_EQ(
      ops_num,
      1,
      common::errors::PreconditionNotMet(
          "graph should only have 2 op nodes, but received %d.", ops_num));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fold_two_squeeze2_fuse_pass);
