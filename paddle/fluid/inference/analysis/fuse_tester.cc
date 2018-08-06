// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/fuse.h"

#include <gtest/gtest.h>

namespace paddle {
namespace inference {
namespace analysis {
namespace fuse {

TEST(Pattern, AddNode) {
  Pattern x;
  ASSERT_EQ(x.pattern_graph_.nodes.size(), 0UL);
  x.AddNode();
  ASSERT_EQ(x.pattern_graph_.nodes.size(), 1UL);
}

TEST(Pattern, AddEdge) {
  Pattern x;
  ASSERT_EQ(x.edges_.size(), 0UL);
  auto* n0 = x.AddNode();
  auto* n1 = x.AddNode();
  x.AddEdge(n0, n1);
  ASSERT_EQ(x.edges_.size(), 1UL);
}

TEST(Pattern, MarkNodesInPattern) {
  Pattern x;
  // consruct the pattern
}

}  // namespace fuse
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
