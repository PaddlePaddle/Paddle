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

#include "paddle/fluid/framework/details/memory_reuse_types.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace framework {
namespace details {

TEST(OrderedNodePairPool, Normal) {
  OrderedNodePairPool pool;
  std::vector<std::unique_ptr<ir::Node>> nodes;

  // clang-format off
  std::vector<std::vector<int64_t>> shapes = {{-1, 10},
                                              {-1, 20},
                                              {1, 2},
                                              {5, 2},
                                              {10, 20},
                                              {-1, 2, 5},
                                              {-1, 1, 5},
                                              {-1, 1}};
  // clang-format on
  const int COUNT = shapes.size();
  ProgramDesc prog;
  BlockDesc* block_desc = prog.MutableBlock(0);
  auto* op_desc = block_desc->AppendOp();
  op_desc->SetType("dummy");
  std::unique_ptr<ir::Node> op = ir::CreateNodeForTest(op_desc);

  for (int i = 0; i < COUNT; ++i) {
    auto desc = block_desc->Var(std::to_string(i));
    desc->SetShape(shapes[i]);
    std::unique_ptr<ir::Node> node = ir::CreateNodeForTest(desc);
    node->inputs.emplace_back(op.get());
    nodes.emplace_back(std::move(node));
  }

  for (auto& node : nodes) {
    pool.Insert(node.get(), op.get());
  }

  // assert its order and interface.
  std::cout << pool.ToString() << std::endl;
  pool.Erase(nodes.front().get());
  std::cout << pool.ToString() << std::endl;

  ASSERT_EQ(pool.size(), static_cast<size_t>(COUNT - 1));
  ASSERT_EQ(pool.GetIndex(nodes.back().get()), 0);

  {
    auto v1 = block_desc->Var("11");
    v1->SetShape({-1, 256, 56, 56});
    std::unique_ptr<ir::Node> node1 = ir::CreateNodeForTest(v1);
    node1->inputs.emplace_back(op.get());
    auto* cache = pool.NodeMatch(node1.get());
    ASSERT_EQ(cache, nullptr);
  }
  {
    auto v2 = block_desc->Var("12");
    v2->SetShape({-1, 2, 5});
    std::unique_ptr<ir::Node> node1 = ir::CreateNodeForTest(v2);
    node1->inputs.emplace_back(op.get());
    auto* cache = pool.NodeMatch(node1.get());
    ASSERT_EQ(pool.GetIndex(cache), 2);  // match 6:[-1,2,5]
  }
  {
    auto v3 = block_desc->Var("13");
    v3->SetShape({2, 5});
    std::unique_ptr<ir::Node> node1 = ir::CreateNodeForTest(v3);
    node1->inputs.emplace_back(op.get());
    auto* cache = pool.NodeMatch(node1.get());
    ASSERT_EQ(pool.GetIndex(cache), 5);  // match  4:[5,2]
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
