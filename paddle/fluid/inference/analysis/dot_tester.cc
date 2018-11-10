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

#include "paddle/fluid/inference/analysis/dot.h"

#include <gtest/gtest.h>
#include <memory>
#include "paddle/fluid/inference/analysis/data_flow_graph.h"

namespace paddle {
namespace inference {
namespace analysis {

class DotTester : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<Dot::Attr> attrs({{"title", "hello"}});
    dot.reset(new Dot(attrs));
    dot->AddNode("a", {Dot::Attr{"shape", "box"}, Dot::Attr("color", "blue")});
    dot->AddNode("b", {});
    dot->AddNode("c", {});
    dot->AddEdge("a", "b", {});
    dot->AddEdge("b", "c", {});
    dot->AddEdge("a", "c", {});
  }

  std::unique_ptr<Dot> dot;
};

TEST_F(DotTester, Build) {
  auto codes = dot->Build();
  // Output the DOT language code, the generated codes are too long to compare
  // the string.
  //
  // The output is
  //
  // digraph G {
  //   title="hello"
  //   node_1
  //   node_2
  //   node_0[label="a" shape="box" color="blue"]
  //   node_0->node_1
  //   node_1->node_2
  //   node_0->node_2
  // } // end G
  LOG(INFO) << '\n' << codes;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
