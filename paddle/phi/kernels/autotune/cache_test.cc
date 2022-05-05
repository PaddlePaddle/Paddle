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

enum ConvAlgos { GEMMKernel = 0, CuDNNKernel_1 = 1, CuDNNKernel_2 = 2 };

TEST(AlgosCache, AlgosCache) {
  auto autotune_cache = phi::autotune::AutoTuneCache::Instance();
  auto& cache = autotune_cache.GetConvForward();

  std::vector<int64_t> x_shape = {4, 224, 224, 3};
  std::vector<int64_t> w_shape = {32, 3, 3, 3};
  std::vector<int> paddings = {0, 0};
  std::vector<int> strides = {2, 2};
  std::vector<int> dilations = {1, 1};
  phi::DataType dtype = paddle::experimental::CppTypeToDataType<float>::Type();

  auto key = phi::autotune::ConvKey(
      x_shape, w_shape, paddings, strides, dilations, dtype);
  EXPECT_EQ(cache.Find(key), false);
  cache.Set(key, ConvAlgos::GEMMKernel);
  EXPECT_EQ(cache.Size(), 1);
  EXPECT_EQ(cache.Find(key), true);
  auto algo = cache.Get(key);
  EXPECT_EQ(algo, ConvAlgos::GEMMKernel);

  x_shape = {4, 128, 128, 3};
  key = phi::autotune::ConvKey(
      x_shape, w_shape, paddings, strides, dilations, dtype);
  EXPECT_EQ(cache.Find(key), false);
  cache.Set(key, ConvAlgos::CuDNNKernel_1);
  EXPECT_EQ(cache.Size(), 2);
  EXPECT_EQ(cache.CacheHits(), 1);
  EXPECT_EQ(cache.CacheMisses(), 2);

  float cache_hit_rate = static_cast<float>(1) / static_cast<float>(3);
  EXPECT_LT(std::abs(cache_hit_rate - cache.CacheHitRate()), 1e-5);

  autotune_cache.UpdateStatus();
  EXPECT_EQ(autotune_cache.Size(), 2);
  EXPECT_EQ(autotune_cache.CacheHits(), 1);
  EXPECT_EQ(autotune_cache.CacheMisses(), 2);
  EXPECT_LT(std::abs(cache_hit_rate - autotune_cache.CacheHitRate()), 1e-5);
}
