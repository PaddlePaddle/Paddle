/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#include <vector>
#include "gtest/gtest.h"

namespace paddle {
namespace distributed {

TEST(BENCHMARK, LargeScaleKV) {
  typedef SparseTableShard<uint64_t, FixedFeatureValue> shard_type;
  shard_type shard;
  uint64_t key = 1;
  auto itr = shard.find(key);
  ASSERT_TRUE(itr == shard.end());

  std::vector<float> vec = {0.0, 0.1, 0.2, 0.3};

  auto& feature_value = shard[key];
  feature_value.resize(vec.size());
  memcpy(feature_value.data(), vec.data(), vec.size() * sizeof(float));

  itr = shard.find(key);
  ASSERT_TRUE(itr != shard.end());

  feature_value = itr.value();
  float* value_data = feature_value.data();

  ASSERT_FLOAT_EQ(value_data[0], 0.0);
  ASSERT_FLOAT_EQ(value_data[1], 0.1);
  ASSERT_FLOAT_EQ(value_data[2], 0.2);
  ASSERT_FLOAT_EQ(value_data[3], 0.3);
}

}  // namespace distributed
}  // namespace paddle
