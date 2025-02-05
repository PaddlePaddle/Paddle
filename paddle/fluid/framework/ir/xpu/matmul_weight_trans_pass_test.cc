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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(MatMulWeightTransPass, basic) {
  Layers layers;

  auto* reshape2_in = layers.data("reshape2_in", {64, 256, 1, 1});
  auto* reshape2_out = layers.reshape2(reshape2_in, std::vector<int>{-1, 256});
  auto* matmul_y = layers.data("matmul_y", {8, 256}, true);
  layers.matmul_v2(reshape2_out, matmul_y, nullptr, false, true);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("matmul_weight_trans_pass");
  VLOG(3) << DebugString(graph);
  pass->Apply(graph.get());
  VLOG(3) << DebugString(graph);

  bool trans_y = true;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "matmul_v2") {
      trans_y = PADDLE_GET_CONST(bool, node->Op()->GetAttr("trans_y"));
    }
  }
  PADDLE_ENFORCE_EQ(
      trans_y,
      false,
      common::errors::PreconditionNotMet(
          "The attribute of matmul_v2 trans_y should be false after pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(matmul_weight_trans_pass);
