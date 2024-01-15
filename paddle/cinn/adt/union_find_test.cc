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

#include <algorithm>

#include "gtest/gtest.h"
#include "paddle/cinn/adt/union_find.h"

namespace cinn::adt::test {

TEST(TestUnion, Naive) {
  UnionFind<int> uf;

  uf.Union(2, 3);
  uf.Union(4, 5);
  uf.Union(3, 4);

  ASSERT_TRUE(uf.IsConnected(2, 3));
  ASSERT_TRUE(uf.IsConnected(2, 4));
  ASSERT_TRUE(uf.IsConnected(2, 5));
  ASSERT_TRUE(uf.IsConnected(3, 4));
  ASSERT_TRUE(uf.IsConnected(3, 5));
  ASSERT_TRUE(uf.IsConnected(4, 5));
  ASSERT_FALSE(uf.IsConnected(1, 2));
  ASSERT_EQ(uf.AllNodeCluster().size(), 1);
  std::vector<int> cluster = uf.AllNodeCluster().at(0);
  std::sort(cluster.begin(), cluster.end());
  ASSERT_EQ(cluster, (std::vector<int>{2, 3, 4, 5}));
  std::vector<int> node_cluster = uf.NodeCluster(5);
  std::sort(node_cluster.begin(), node_cluster.end());
  ASSERT_EQ(node_cluster, cluster);
}

}  // namespace cinn::adt::test
