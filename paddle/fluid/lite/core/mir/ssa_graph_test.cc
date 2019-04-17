// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/mir/ssa_graph.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/mir/passes.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/program_fake_utils.h"

namespace paddle {
namespace lite {
namespace mir {

void BuildFc(framework::ProgramDesc* desc, const std::string& x,
             const std::string& w, const std::string& b,
             const std::string& out) {
  auto* fc = desc->MutableBlock(0)->AppendOp();
  fc->SetInput("Input", {x});
  fc->SetInput("W", {w});
  fc->SetInput("Bias", {b});
  fc->SetOutput("Out", {out});
}

TEST(SSAGraph, test) {
  auto program = FakeProgram();
  SSAGraph graph;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};

  graph.Build(program, places);

  Visualize(&graph);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_LITE_KERNEL(fc, kHost, kFloat);
