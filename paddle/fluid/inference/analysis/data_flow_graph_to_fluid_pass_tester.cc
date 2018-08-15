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

#include "paddle/fluid/inference/analysis/data_flow_graph_to_fluid_pass.h"

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/io.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST_F(DFG_Tester, Test) {
  DataFlowGraph graph;

  FluidToDataFlowGraphPass pass0;
  DataFlowGraphToFluidPass pass1;
  ASSERT_TRUE(pass0.Initialize(&argument));
  ASSERT_TRUE(pass1.Initialize(&argument));

  pass0.Run(&graph);
  pass1.Run(&graph);

  pass0.Finalize();
  pass1.Finalize();

  LOG(INFO) << graph.nodes.size();
}

};  // namespace analysis
};  // namespace inference
};  // namespace paddle
