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

TEST(StackFusePass, basic) {
  Layers layers;
  auto* block = layers.Block();

  auto* stack_x = layers.data("stack_x", {-1, 64, 64});
  auto* stack_out = layers.stack({stack_x, stack_x, stack_x}, 1);
  stack_out->SetShape({-1, 3, 64, 64});
  auto* add_x = layers.data("add_x", {-1, 24, 64, 64});
  layers.elementwise_add(add_x, stack_out);

  OpDesc* fused_multi_transformer_op = block->AppendOp();
  fused_multi_transformer_op->SetType("fused_multi_transformer");
  fused_multi_transformer_op->SetInput("SrcMask", {stack_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("stack_fuse_pass");
  pass->Apply(graph.get());
  auto stack_num = GetNumOpNodes(graph, "stack");
  PADDLE_ENFORCE_EQ(stack_num,
                    0,
                    common::errors::PreconditionNotMet(
                        "stack op should be removed from graph, but graph "
                        "still has %d stack op.",
                        stack_num));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(stack_fuse_pass);
