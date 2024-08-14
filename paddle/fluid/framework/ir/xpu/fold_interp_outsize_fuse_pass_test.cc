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

TEST(FoldInterpOutsizeFusePass, basic) {
  Layers layers;
  auto* block = layers.Block();

  auto* shape_x = layers.data("shape_x", {1, 18, 288, 288});
  auto* concat_y =
      layers.data("concat_y", {576, 576}, true, proto::VarType::INT64);
  auto* shape_out = layers.shape(shape_x);
  auto* cast1_out = layers.cast(shape_out, 2, 3);
  auto* slice_out = layers.slice(cast1_out, {0}, {0}, {2});
  auto* concat_out = layers.concat({slice_out, concat_y}, 0);
  auto split_outs = layers.split(concat_out, 0, 0, {2, 2});
  auto* split_out_1 = split_outs[1];
  auto* cast2_out = layers.cast(split_out_1, 3, 2);

  OpDesc* bilinear_interp_v2_op = block->AppendOp();
  bilinear_interp_v2_op->SetType("bilinear_interp_v2");
  bilinear_interp_v2_op->SetInput("X", {shape_x->Name()});
  bilinear_interp_v2_op->SetInput("OutSize", {cast2_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("fold_interp_outsize_fuse_pass");
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

USE_PASS(fold_interp_outsize_fuse_pass);
