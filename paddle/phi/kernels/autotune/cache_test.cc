// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/autotune/cache.h"
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include "glog/logging.h"

void Algo() { VLOG(3) << "algo test"; }

TEST(AlgosCache, AlgosCache) {
  phi::autotune::AlgorithmsCache<std::function<void()>> cache;
  std::vector<int64_t> x_shape = {4, 224, 224, 3};
  std::vector<int64_t> w_shape = {32, 3, 3, 3};
  std::vector<int> paddings = {0, 0};
  std::vector<int> strides = {2, 2};
  std::vector<int> dilations = {1, 1};
  phi::DataType dtype = paddle::experimental::CppTypeToDataType<float>::Type();

  auto key =
      cache.ConvKey(x_shape, w_shape, paddings, strides, dilations, dtype);
  EXPECT_EQ(cache.Find(key), false);
  cache.Set(key, Algo);
  EXPECT_EQ(cache.Find(key), true);
  auto algo = cache.Get(key);
  algo();

  x_shape = {4, 128, 128, 3};
  key = cache.ConvKey(x_shape, w_shape, paddings, strides, dilations, dtype);
  EXPECT_EQ(cache.Find(key), false);
  float cache_hit_rate = static_cast<float>(1) / static_cast<float>(3);
  EXPECT_LT(std::abs(cache_hit_rate - cache.CacheHitRate()), 1e-5);
}
