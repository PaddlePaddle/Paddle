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

TEST(CastMixedPrecisionOpFusePass, cast_before) {
  Layers layers;
  auto* block = layers.Block();

  auto* cast_in = layers.data("cast_in");
  auto* cast_out = layers.cast(cast_in, 5, 4);
  OpDesc* conv2d_xpu = block->AppendOp();
  conv2d_xpu->SetType("conv2d_xpu");
  conv2d_xpu->SetInput("x", {cast_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("cast_mixed_precision_op_fuse_pass");
  pass->Apply(graph.get());
  auto num = GetNumOpNodes(graph, "cast");
  PADDLE_ENFORCE_EQ(
      num,
      0,
      common::errors::PreconditionNotMet(
          "cast op should be removed from graph, but graph still has %d ops.",
          num));
}

TEST(CastMixedPrecisionOpFusePass, cast_after) {
  Layers layers;
  auto* block = layers.Block();

  auto* cast_in = layers.data("cast_in");
  OpDesc* conv2d_xpu = block->AppendOp();
  conv2d_xpu->SetType("conv2d_xpu");
  conv2d_xpu->SetOutput("out", {cast_in->Name()});
  layers.cast(cast_in, 4, 5);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("cast_mixed_precision_op_fuse_pass");
  pass->Apply(graph.get());
  auto num = GetNumOpNodes(graph, "cast");
  PADDLE_ENFORCE_EQ(
      num,
      0,
      common::errors::PreconditionNotMet(
          "cast op should be removed from graph, but graph still has %d ops.",
          num));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cast_mixed_precision_op_fuse_pass);
