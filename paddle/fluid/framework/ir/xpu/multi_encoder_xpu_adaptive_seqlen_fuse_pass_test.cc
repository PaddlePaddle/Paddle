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

TEST(MultiEncoderXPUAdaptiveSeqlenFusePass, V1) {
  Layers layers;
  auto* block = layers.Block();

  auto* embedding_xpu_out = layers.data("embedding_xpu_out");
  OpDesc* embedding_xpu = block->AppendOp();
  embedding_xpu->SetType("embedding_with_eltwise_add_xpu");
  embedding_xpu->SetOutput("out", {embedding_xpu_out->Name()});
  auto* layer_norm_out = layers.layer_norm(embedding_xpu_out)[0];

  auto* mask = layers.data("mask");
  auto* matmul_out = layers.matmul(mask, mask);
  auto* scale_out = layers.scale(matmul_out);
  auto* stack_out = layers.stack({scale_out, scale_out});

  OpDesc* multi_encoder_xpu = block->AppendOp();
  multi_encoder_xpu->SetType("multi_encoder_xpu");
  multi_encoder_xpu->SetInput("x", {layer_norm_out->Name()});
  multi_encoder_xpu->SetInput("mask", {stack_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get(
      "multi_encoder_xpu_adaptive_seqlen_fuse_pass");
  pass->Apply(graph.get());
  auto num = GetNumOpNodes(graph, "matmul") + GetNumOpNodes(graph, "scale") +
             GetNumOpNodes(graph, "stack");
  PADDLE_ENFORCE_EQ(
      num,
      0,
      common::errors::PreconditionNotMet(
          "matmul/scale/stack ops should be removed from graph, but graph "
          "still has %d ops.",
          num));
}

TEST(MultiEncoderXPUAdaptiveSeqlenFusePass, V2) {
  Layers layers;
  auto* block = layers.Block();

  auto* embedding_xpu_out = layers.data("embedding_xpu_out");
  OpDesc* embedding_xpu = block->AppendOp();
  embedding_xpu->SetType("embedding_with_eltwise_add_xpu");
  embedding_xpu->SetOutput("out", {embedding_xpu_out->Name()});
  auto* layer_norm_out = layers.layer_norm(embedding_xpu_out)[0];

  auto* mask = layers.data("mask");
  auto* not_equal_y = layers.data("not_equal_y");
  auto* not_equal_out = layers.not_equal(mask, not_equal_y);
  auto* cast_out = layers.cast(not_equal_out);
  auto* unsqueeze_0_out = layers.unsqueeze2(cast_out);
  auto* matmul_out = layers.matmul_v2(unsqueeze_0_out, unsqueeze_0_out);
  auto* scale_0_out = layers.scale(matmul_out);
  auto* scale_1_out = layers.scale(scale_0_out);
  auto* unsqueeze_1_out = layers.unsqueeze2(scale_1_out);
  auto* tile_out = layers.tile(unsqueeze_1_out);

  OpDesc* multi_encoder_xpu = block->AppendOp();
  multi_encoder_xpu->SetType("multi_encoder_xpu");
  multi_encoder_xpu->SetInput("x", {layer_norm_out->Name()});
  multi_encoder_xpu->SetInput("mask", {tile_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get(
      "multi_encoder_xpu_adaptive_seqlen_fuse_pass");
  pass->Apply(graph.get());
  auto num = GetNumOpNodes(graph, "not_equal") + GetNumOpNodes(graph, "cast") +
             GetNumOpNodes(graph, "unsqueeze2") +
             GetNumOpNodes(graph, "matmul_v2") + GetNumOpNodes(graph, "scale") +
             GetNumOpNodes(graph, "tile");
  PADDLE_ENFORCE_EQ(num,
                    0,
                    common::errors::PreconditionNotMet(
                        "not_equal/cast/unsqueeze2/matmul_v2/scale ops should "
                        "be removed from graph, but graph "
                        "still has %d ops.",
                        num));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(multi_encoder_xpu_adaptive_seqlen_fuse_pass);
