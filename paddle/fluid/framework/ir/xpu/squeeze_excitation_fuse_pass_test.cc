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

TEST(SqueezeExcitationFusePass, V1) {
  Layers layers;
  auto* block = layers.Block();

  auto* pool2d_inp = layers.data("pool2d_inp", {1, 24, 14, 14});
  auto* pool2d_out = layers.pool2d(pool2d_inp, false);

  auto* conv2d_xpu_op1_out = layers.data("conv2d_xpu_op1_out");
  OpDesc* conv2d_xpu_op1 = block->AppendOp();
  conv2d_xpu_op1->SetType("conv2d_xpu");
  conv2d_xpu_op1->SetInput("x", {pool2d_out->Name()});
  conv2d_xpu_op1->SetOutput("out", {conv2d_xpu_op1_out->Name()});

  auto* conv2d_xpu_op2_out = layers.data("conv2d_xpu_op2_out");
  OpDesc* conv2d_xpu_op2 = block->AppendOp();
  conv2d_xpu_op2->SetType("conv2d_xpu");
  conv2d_xpu_op2->SetInput("x", {conv2d_xpu_op1_out->Name()});
  conv2d_xpu_op2->SetOutput("out", {conv2d_xpu_op2_out->Name()});

  layers.elementwise_mul(pool2d_inp, conv2d_xpu_op2_out);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("squeeze_excitation_fuse_pass");
  pass->Apply(graph.get());
  auto num = GetNumOpNodes(graph, "pool2d") +
             GetNumOpNodes(graph, "conv2d_xpu") +
             GetNumOpNodes(graph, "elementwise_mul");
  PADDLE_ENFORCE_EQ(num,
                    0,
                    common::errors::PreconditionNotMet(
                        "pool2d/conv2d_xpu/elementwise_mul ops should be "
                        "removed from graph, but graph "
                        "still has %d ops. ",
                        num));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(squeeze_excitation_fuse_pass);
