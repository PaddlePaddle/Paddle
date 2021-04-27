// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"
#include <gtest/gtest.h>
#include "paddle/fluid/distributed/index_dataset/index_dataset.pb.h"
using paddle::distributed::GraphIndex;

TEST(GRAPH_INDEX, RUN) {
  GraphIndex graph1, graph2;
  int width = 3, height = 5, path_num = 9;
  graph1.set_width(width);
  graph1.set_height(height);
  graph1.set_item_path_nums(path_num);
  graph1.add_item(4, {3, 54, 12});
  graph1.add_item(6, {7, 2, 100});
  graph1.add_item(34, {7, 8, 12});
  graph1.save("paddle_graph_index.param");
  graph2.load("paddle_graph_index.param");
  ASSERT_EQ(graph2.width(), width);
  ASSERT_EQ(graph2.height(), height);
  auto map_2 = graph2.get_item_path_dict();
  auto map_1 = graph1.get_item_path_dict();
  ASSERT_EQ(map_2.size(), map_1.size());
  for (auto p1 : map_1) {
    ASSERT_EQ(map_2.find(p1.first) != map_2.end(), true);
    auto vec = map_2[p1.first];
    ASSERT_EQ(vec.size(), p1.second.size());
    for (size_t i = 0; i < vec.size(); i++) {
      ASSERT_EQ(vec[i], p1.second[i]);
    }
  }
}