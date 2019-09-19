/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/fusion_group_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(FusionGroupPass, elementwise_2) {
  Layers layers;
  auto* x = layers.data("x", {16, 16});
  auto* y = layers.data("y", {16, 32});
  auto* mul_out = layers.mul(x, y);
  mul_out->SetShape({16, 32});
  auto* z = layers.data("z", {16, 32});
  auto* add_out = layers.elementwise_add(mul_out, z);
  auto* relu_out = layers.relu(add_out);
  relu_out->SetShape({16, 32});
  auto* w = layers.data("w", {16, 32});
  layers.elementwise_add(relu_out, w);

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("fusion_group_pass");
  LOG(INFO) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  LOG(INFO) << DebugString(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fusion_group_pass);
