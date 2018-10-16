/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/inference/analysis/data_flow_graph_to_fluid_pass.h"
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"
#include "paddle/fluid/inference/analysis/pass_manager.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

class TestDfgPassManager final : public DfgPassManager {
 public:
  TestDfgPassManager() = default;
  virtual ~TestDfgPassManager() = default;
  // Short identifier.
  std::string repr() const override { return "test-pass-manager"; }
  // Long description.
  std::string description() const override { return "test doc"; }
};

TEST(PassManager, DFG_pass_manager) {
  TestDfgPassManager manager;
  DFG_GraphvizDrawPass::Config config("./", "dfg.dot");

  manager.Register("fluid-to-flow-graph", new FluidToDataFlowGraphPass);
  manager.Register("graphviz", new DFG_GraphvizDrawPass(config));
  manager.Register("dfg-to-fluid", new DataFlowGraphToFluidPass);

  Argument argument(FLAGS_inference_model_dir);

  ASSERT_TRUE(&argument);
  ASSERT_TRUE(manager.Initialize(&argument));
  manager.RunAll();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
