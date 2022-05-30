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

#include "paddle/fluid/framework/ir/adaptive_pool2d_convert_global_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(AdaptivePool2dConvertGlobalPass, basic) {
  Layers layers;
  auto* x = layers.data("x", {1, 92, 28, 28});
  AttributeMap attrs;
  attrs["adaptive"] = true;
  attrs["ksize"] = std::vector<int>{1, 1};
  attrs["pooling_type"] =
      std::string("avg");  // adaptive has no effect on max pooling
  layers.pool2d(x, false, &attrs);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass =
      PassRegistry::Instance().Get("adaptive_pool2d_convert_global_pass");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  VLOG(3) << DebugString(graph);

  bool global_pooling = false;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "pool2d") {
      if (node->Op()->HasAttr("global_pooling")) {
        global_pooling =
            BOOST_GET_CONST(bool, node->Op()->GetAttr("global_pooling"));
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      global_pooling, true,
      platform::errors::PreconditionNotMet(
          "The attribute of pool2d global_pooling should be true after fuse"));
}

TEST(AdaptivePool2dConvertGlobalPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("adaptive_pool2d_convert_global_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(adaptive_pool2d_convert_global_pass);
