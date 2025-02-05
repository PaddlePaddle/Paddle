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

#include <string>

#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

AdaptivePool2dConvertGlobalPass::AdaptivePool2dConvertGlobalPass() {  // NOLINT
  AddOpCompat(OpCompat("pool2d"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("pooling_type")
      .IsStringIn({"max", "avg"})
      .End()
      .AddAttr("ksize")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("global_pooling")
      .IsBoolEQ(true)
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("exclusive")
      .IsType<bool>()
      .End()
      .AddAttr("adaptive")
      .IsBoolEQ(false)
      .End()
      .AddAttr("ceil_mode")
      .IsType<bool>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NHWC", "NCHW"})
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End();
}

void AdaptivePool2dConvertGlobalPass::ApplyImpl(ir::Graph* graph) const {
  std::string name_scope = "adaptive_pool2d_convert_global_pass";

  FusePassBase::Init(name_scope, graph);
  int num = 0;
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->Type() == "pool2d" && op->HasAttr("adaptive") &&
          op->HasAttr("ksize")) {
        if (op->HasAttr("global_pooling")) {
          bool global_pooling =
              PADDLE_GET_CONST(bool, op->GetAttr("global_pooling"));
          if (global_pooling) continue;
        }
        if (!op->HasAttr("pooling_type")) continue;
        std::string type =
            PADDLE_GET_CONST(std::string, op->GetAttr("pooling_type"));
        // adaptive has no effect on max pooling
        if (type == "max") continue;
        bool adaptive = PADDLE_GET_CONST(bool, op->GetAttr("adaptive"));
        std::vector<int> ksize =
            PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
        if (adaptive && ksize.size() == 2 && ksize[0] == 1 && ksize[1] == 1) {
          op->SetAttr("adaptive", false);
          op->SetAttr("global_pooling", true);
          ++num;
        }
      }
    }
  }
  AddStatis(num);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(adaptive_pool2d_convert_global_pass,
              paddle::framework::ir::AdaptivePool2dConvertGlobalPass);

REGISTER_PASS_CAPABILITY(adaptive_pool2d_convert_global_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "pool2d", 0));
