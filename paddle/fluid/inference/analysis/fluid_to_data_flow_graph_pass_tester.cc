//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST(FluidToDataFlowGraphPass, Test) {
  FluidToDataFlowGraphPass pass;
  Argument argument(FLAGS_inference_model_dir);
  pass.Initialize(&argument);
  pass.Run(argument.main_dfg.get());
  // Analysis is sensitive to ProgramDesc, careful to change the original model.
  ASSERT_EQ(argument.main_dfg->nodes.size(), 38UL);
  pass.Finalize();
  ASSERT_FALSE(argument.main_dfg->DotString().empty());
  EXPECT_FALSE(argument.main_dfg->inputs().empty());
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
