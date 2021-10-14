/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <ThreadPool.h>

#include <unistd.h>
#include <string>
#include <vector>
#include <thread>  // NOLINT

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/table/depends/ctr_large_scale_kv.h"

namespace paddle {
namespace distributed {

TEST(BENCHMARK, LargeScaleKV) {
  std::shared_ptr<CtrValueBlock> shard = std::make_shared<CtrValueBlock>();
  uint64_t key = 1;
  auto itr = shard->Find(key);
  ASSERT_TRUE(itr == shard->end());

  std::vector<float> vec = {0.0, 0.1, 0.2, 0.3};

  auto* feature_value = shard->Init(key);
  feature_value->resize(vec.size());
  memcpy(const_cast<float*>(feature_value->data()), vec.data(), vec.size() * sizeof(float));

  itr = shard->Find(key);
  ASSERT_TRUE(itr != shard->end());
  
  feature_value = itr->second;
  float* value_data = const_cast<float*>(feature_value->data());

  ASSERT_FLOAT_EQ(value_data[0], 0.0);
  ASSERT_FLOAT_EQ(value_data[1], 0.1);
  ASSERT_FLOAT_EQ(value_data[2], 0.2);
  ASSERT_FLOAT_EQ(value_data[3], 0.3);

}

}  // namespace distributed
}  // namespace paddle
