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

#include <map>
#include <unordered_set>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_cache_key.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

TEST(CinnCacheKeyTest, TestAsUnorderedKey) {
  std::unordered_set<CinnCacheKey, CinnCacheKey::Hash> test_set;

  ProgramDesc empty_program;
  ir::Graph empty_graph(empty_program);

  ProgramDesc program;
  auto *global_block = program.MutableBlock(0);
  auto *x = global_block->Var("X");
  x->SetType(proto::VarType::LOD_TENSOR);
  ir::Graph graph(program);

  LoDTensor tensor;
  tensor.Resize({1, 2, 3});
  const LoDTensor *tensor_pointer = &tensor;
  std::map<std::string, const LoDTensor *> feed_tensors = {
      {"X", tensor_pointer}};

  DDim ddim = paddle::framework::make_ddim({1, 2, 3});
  std::map<std::string, DDim> feed_shapes = {{"X", ddim}};

  CinnCacheKey cache_key1(empty_graph, feed_tensors);
  CinnCacheKey cache_key2(empty_graph, feed_shapes);
  EXPECT_EQ(cache_key1, cache_key2);

  CinnCacheKey cache_key3(graph, feed_shapes);
  CinnCacheKey cache_key4(graph, feed_tensors);
  EXPECT_EQ(cache_key3, cache_key4);

  CinnCacheKey cache_key5(empty_graph,
                          std::map<std::string, const LoDTensor *>());
  CinnCacheKey cache_key6(empty_graph, std::map<std::string, DDim>());
  EXPECT_EQ(cache_key5, cache_key6);

  EXPECT_NE(cache_key1, cache_key3);
  EXPECT_NE(cache_key4, cache_key2);

  EXPECT_NE(cache_key3, cache_key5);
  EXPECT_NE(cache_key6, cache_key4);

  EXPECT_NE(cache_key5, cache_key1);
  EXPECT_NE(cache_key2, cache_key6);

  test_set.insert(cache_key1);
  test_set.insert(cache_key2);
  test_set.insert(cache_key3);
  test_set.insert(cache_key4);
  test_set.insert(cache_key5);
  test_set.insert(cache_key6);
  EXPECT_EQ(test_set.size(), 3U);

  auto iter = test_set.find(cache_key1);
  EXPECT_NE(iter, test_set.end());
  test_set.erase(iter);
  EXPECT_EQ(test_set.size(), 2U);
  EXPECT_EQ(test_set.find(cache_key2), test_set.end());

  iter = test_set.find(cache_key3);
  EXPECT_NE(iter, test_set.end());
  test_set.erase(iter);
  EXPECT_EQ(test_set.size(), 1U);
  EXPECT_EQ(test_set.find(cache_key4), test_set.end());

  iter = test_set.find(cache_key5);
  EXPECT_NE(iter, test_set.end());
  test_set.erase(iter);
  EXPECT_EQ(test_set.size(), 0U);
  EXPECT_EQ(test_set.find(cache_key6), test_set.end());
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
