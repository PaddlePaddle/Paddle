/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_shard.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {

TEST(DataShard, Read) {
  int64_t shard_id = 1;
  int64_t shard_size = 10;
  DataShard shard = DataShard(shard_id, shard_size);

  std::vector<std::pair<int64_t, int64_t>> ids = {
      {4, 0}, {3, 2}, {2, 1}, {1, 3}};
  std::vector<int64_t> indexs(ids.size());
  shard.GetIndexsByIds(ids, &indexs, true).wait();

  ASSERT_EQ(indexs[0], 10);
  ASSERT_EQ(indexs[1], 12);
  ASSERT_EQ(indexs[2], 11);
  ASSERT_EQ(indexs[3], 13);

  std::vector<std::pair<int64_t, int64_t>> ids1 = {
      {4, 0}, {3, 2}, {6, 1}, {4, 3}};
  std::vector<int64_t> indexs1(ids1.size());
  shard.GetIndexsByIds(ids1, &indexs1, true).wait();

  ASSERT_EQ(indexs1[0], 10);
  ASSERT_EQ(indexs1[1], 14);
  ASSERT_EQ(indexs1[2], 11);
  ASSERT_EQ(indexs1[3], 10);
}

}  // namespace framework
}  // namespace paddle
