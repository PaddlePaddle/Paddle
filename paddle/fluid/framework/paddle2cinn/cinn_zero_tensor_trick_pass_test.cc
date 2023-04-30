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

#include "paddle/fluid/framework/paddle2cinn/cinn_zero_tensor_trick_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

TEST(CinnZeroTensorTrickPass, basic) {
  ir::Layers layers;
  auto* x = layers.data("x", {});
  auto* y = layers.data("y", {3, 4});
  auto* add_out_0 = layers.elementwise_add(x, y, nullptr, 0);
  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = ir::PassRegistry::Instance().Get("cinn_zero_tensor_trick_pass");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  VLOG(3) << DebugString(graph);

  for (auto* n : graph->Nodes()) {
    if (n->IsVar()) {
      if (n->Var() && n->Var()->GetType() == proto::VarType::LOD_TENSOR) {
        std::vector<int64_t> shape = n->Var()->GetShape();
        PADDLE_ENFORCE_EQ(
            shape.empty(),
            false,
            platform::errors::PreconditionNotMet(
                "The shape of elementwise_add should not be empty after fuse"));
      }
    }
  }
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

USE_PASS(cinn_zero_tensor_trick_pass);
