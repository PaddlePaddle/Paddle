/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/ir/unsqueeze2_eltwise_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(UnsqueezeEltwiseFusePass, basic) {
  Layers layers;
  auto* x = layers.data("x", {1, 92, 28, 28});
  auto* y = layers.data("y", {1, 92});
  std::vector<int> axes{2, 3};
  auto* unsqz_out = layers.unsqueeze2(y, axes);
  AttributeMap attrs;
  attrs["axis"] = -1;
  layers.elementwise_mul(x, unsqz_out, nullptr, &attrs);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("unsqueeze2_eltwise_fuse_pass");
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fused_nodes_after = GetNumOpNodes(graph, "elementwise_mul");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after + 2,
                    platform::errors::PreconditionNotMet(
                        "The number of nodes before and after the fuse does "
                        "not meet expectations"));
  PADDLE_ENFORCE_EQ(
      num_fused_nodes_after,
      1,
      platform::errors::PreconditionNotMet(
          "The number of fusion nodes does not meet expectations after fuse"));
}

TEST(UnsqueezeEltwiseFusePass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("unsqueeze2_eltwise_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(unsqueeze2_eltwise_fuse_pass);
