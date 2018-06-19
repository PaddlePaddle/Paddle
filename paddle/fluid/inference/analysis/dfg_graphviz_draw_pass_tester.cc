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

#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST_F(DFG_Tester, dfg_graphviz_draw_pass_tester) {
  auto dfg = ProgramDescToDFG(*argument.origin_program_desc);
  DFG_GraphvizDrawPass::Config config("./", "test");
  DFG_GraphvizDrawPass pass(config);
  pass.Initialize(&argument);
  pass.Run(&dfg);

  // test content
  std::ifstream file("./graph_test.dot");
  ASSERT_TRUE(file.is_open());

  std::string line;
  int no{0};
  while (std::getline(file, line)) {
    no++;
  }
  // DFG is sensitive to ProgramDesc, be careful to change the existing models.
  ASSERT_EQ(no, 112);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
